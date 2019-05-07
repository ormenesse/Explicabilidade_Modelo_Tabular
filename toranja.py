from io import BytesIO
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, lars_path
from sklearn.metrics import pairwise_distances
from sklearn.metrics import silhouette_score
from sklearn.utils import check_random_state
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import seaborn as sns
import lightgbm as lgb

class toranja(object):
    
    """
       #Classe feita para avaliarmos a explicabilidade agrupada de um modelo.#
       Args:
          X: Base de treinamento.
          cat_cols: colunas categóricas em números.
          modelo: Modelo que será explicado (deve conter método predict_proba())
          random_number: número que será utilizado em algoritmos pseudo aleatórios
       Returns:
           Arquivo planilha Excel com dados consolidados.
           Apenas os coeficientes da regressão.
           Função clusterizada para fácil interpretabilidade de modelos.
        Author:
           Vinícius Ormenesse
    """
    
    def __init__(self,X,modelo,cat_cols=[],random_number=4242):
        assert str(type(X)) == "<class 'pandas.core.frame.DataFrame'>", 'Por favor, inserir uma base no formato Pandas.'
        for c in cat_cols:
            assert 'str' not in str(type(c)),'Lista de categóricas deve ser apenas com os valores dos índices das colunas.'
        self.colunas = list(X.columns)
        self.categorical_features = cat_cols
        self.norm = self.__min_max_norm(X,cat_cols)
        self.modelo = modelo
        self.feature_freq = self.__feature_frequencies(X,self.categorical_features)
        self.random_number = random_number
        #contando quantos missings eu tenho em cada coluna
        self.missings_counts = 1-X.describe().transpose()['count']/X.shape[0]
        self.colunas_missing = []
        self.norm_kmeans = {}
        self.kmeans = None
        #Ajustar modelo de árvore para que método fique mais produtivo
        self.tree_model_kmeans = None 
        self.pd_exp = pd.DataFrame([])

    def __min_max_norm(self,X,categorical_columns):
        norm = {}
        for col in X.columns:
            if col in categorical_columns:
                norm[col] = [X[col].min(),X[col].max(),1,0]
            else:
                norm[col] = [X[col].min(),X[col].max(),X[col].std(),X[col].mean()]
        return norm

    def __feature_frequencies(self,X,categorical_columns):
        norm = {}
        for column in categorical_columns:
            norm[column] = X[X.columns[column]].value_counts(dropna=False,normalize=True).sort_index()
        return norm

    """
        Gera variáveis aleátorias para criação da regressão logística, em volta da amostra original.
        -data retorna dado gerado para treinamento da regressão, caso existam variáveis categóricas, esta será maracada como zero ou 1 apenas para saber se é a mesma variável da amostra ou não.
        -inverse é um vetor igual como seria uma amostra real, apenas que criado sintéticamente
        -inverse = data <> Diferença é que inverse usaremos para predict_proba e data para regressão.
    """
    def __data_inverse(self,
                       data_row,
                       colunas,
                       num_samples
                       ):
        
        random_state = check_random_state(self.random_number)
        data = np.zeros((num_samples, data_row.shape[0]))
        #categorical_features = range(data_row.shape[0])
        data = random_state.normal(
                0, 1, num_samples * data_row.shape[0]).reshape(
                num_samples, data_row.shape[0])
        #working with nan data, randomly 
        data_nan = np.isnan(np.zeros((num_samples, data_row.shape[0]))+data_row)*np.random.choice([True,False],(num_samples, data_row.shape[0]))+0
        data_nan = data_nan.astype('float64')
        data_nan[data_nan == 1] = np.nan
        data_nan = data_nan+np.nan_to_num(data_row)
        #
        data = data * np.array([self.norm[c][2] for c in self.colunas]) + data_nan #data_row
        data[0] = data_row.copy()
        inverse = data.copy()
        first_row = data_row
        for column in self.categorical_features:
            #trabalhando com a base de treinamento para explicar o modelo
            feature_frequencies = self.feature_freq[column]
            feature_values = list(feature_frequencies.index)
            values = feature_values
            freqs = feature_frequencies
            inverse_column = random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = np.array([1 if x == first_row[column]
                                      else 0 for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        inverse[0] = data_row
        #retorna os dados necessários
        return data, inverse
        
    """
        Definição de distância para criação de pesos na regressão. Queremos que a linha real seja a mais importante na geração dos Betas da regressão
    """
    def kernel(self,d, kernel_width):
        return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))
    
    """
        Regressão - Função interna
        No caso, esse método funciona tanto para explicabilidade de classificação como explicabilidade de regressão.
    """
    def __return_best_coefficients(self,value,n_samples):
        data, inverse = self.__data_inverse(value,self.colunas,n_samples)
        #voltando os valores para que eu possa encontrar suas respectivas probabilidades de classe
        inv_df = pd.DataFrame(inverse.copy(),columns=self.colunas)
        for i,col in enumerate(self.colunas):
            if col not in self.categorical_features:
                inv_df[col] = inv_df[col]*(self.norm[col][1]-self.norm[col][0])+self.norm[col][0]
        #normalizando os dados que vão para regressão
        data_df = pd.DataFrame(data.copy(),columns=self.colunas)
        for i,col in enumerate(self.colunas):
            if col not in self.categorical_features:
                data_df[col]=(data_df[col]-self.norm[col][0])/(self.norm[col][1]-self.norm[col][0])
        #colocando os misssing em seus respectivos lugares
        for col in self.colunas_missing:
            data_df[col+'_NAN'] = 0
            data_df[col+'_NAN'][inv_df[col].isnull()] = 1
        #eu tenho que assinalar o nan para algum número
        data_df.fillna(0,inplace=True)
        #predizendo os resultados
        predict_generated = self.modelo.predict_proba(inv_df)
        #calculando distancias para pesos da regressão
        distances = pairwise_distances(
                    data_df.values,
                    data_df.values[0].reshape(1, -1),
                    metric='euclidean'
            ).ravel()
        #isso aqui poderia estar em uma variável global.
        kernel_width = np.sqrt(data_df.shape[1]) * .75
        #
        distances_kernel = self.kernel(distances,kernel_width)
        #forçando a variável inicial ser a mais importante
        distances_kernel[0] = 3
        distances = distances.round(2)
        #fitando modelo
        model_regressor = Ridge(alpha=1, fit_intercept=True,random_state=4242)
        model_regressor.fit(data_df,predict_generated[:,1],sample_weight=distances_kernel)
        #retornando o que realmente me importa
        return model_regressor.coef_
    
    def tabular_analysis(self,X,n_samples=500,sample_base=0.1,n_clusters=20,missing_perc=0.20,nome_arquivo='resultado_explicabilidade'):
        """
            Args:
                X: Base que será utilizada para explicabilidade.
                n_samples: Quantidade de amostras para regressão Ridge.
                sample_base:Amostragem da base de entrada X. Não é aconselhável utilizar toda a base de treinamento. Tempo de execução é muito alto.
                n_clusters: Quantidade de Clusters criados no método do cotovelo.
                missing_perc: Valor entre 0 e 1. Onde 0 significa não olhar por missings e 1 olhar todas as colunas com missings. Método aceita modelos que aceitam missings, então missing perc avaliará importância dos missings no modelo.
                nome_arquivo: Nome do arquivo xlsx de saída.
            Returns:
                Arquivo xlsx com informações dos clusters criados para interpretabilidade.
        """
        assert missing_perc <= 1 and missing_perc >= 0, 'Valor missing_perc deve estar entre 0 e 1.'
        assert sample_base <= 1 and sample_base >= 0, 'A variável sample_base deve ser um valor entre 0 e 1'
        print('Initializing Cluster Interpretable Model-agnostic Explanations...')
        #amostrando a base
        x_samples = X.sample(frac=sample_base)
        x_samples.replace([np.nan,np.NAN,np.NaN],[np.nan,np.nan,np.nan],inplace=True)
        #colocando coluna missing nas amostras
        if self.colunas_missing == []:
            for col in self.missings_counts.index:
                if self.missings_counts.loc[col] >= missing_perc:
                    self.colunas_missing.append(col)
        #trabalhando com análise de cada amostra.
        for i in tqdm(range(0,x_samples.shape[0])):
            if i == 0:
                first_return = self.__return_best_coefficients(x_samples.iloc[i].values,n_samples)
                #print(first_return.shape)
                explanations = np.zeros((x_samples.shape[0],x_samples.shape[1]+len(self.colunas_missing)))
                explanations[0,:] = first_return
            else:
                explanations[i,:] = self.__return_best_coefficients(x_samples.iloc[i].values,n_samples)
        probs = pd.DataFrame(self.modelo.predict_proba(x_samples),columns=['Prob_0','Prob_1'])
        
        #Criando um DataFrame com todos meus coeficientes:
        colsmis = []
        for cols in self.colunas_missing:
            colsmis.append(cols+'_nan')
        colunas_exp1 = self.colunas+colsmis
        exp1 = pd.DataFrame(explanations,columns=colunas_exp1)
        
        #normalizando colunas
        norm = {}
        for col in exp1.columns:
            norm[col] = [exp1[col].min(),exp1[col].max()]
            exp1[col]=(exp1[col]-exp1[col].min())/(exp1[col].max()-exp1[col].min())
        #salvando
        self.norm_kmeans = norm
        #colocando a probabilidade junto a tabela.
        """
        #antes eu tinha pensado em trabalhar com clusterização conjunta, mas agora eu não acho que valha a pena.
        exp1 = pd.concat([exp1,probs['Prob_1']],axis=1)
        """
        exp1 = exp1.fillna(0)
        self.pd_exp = exp1
        #Clusterizando
        print('Testando diversos agrupamentos.')
        dists = []
        distortions = []
        sil_coeff = []
        #K = range(2,10)
        K = range(2,n_clusters+1)
        for k in K:
            kmeans_teste = KMeans(n_clusters=k,random_state=np.random.randint(1, 1000 + 1),n_init=20).fit(exp1)
            temp = exp1.copy()
            temp['cluster'] = pd.DataFrame(kmeans_teste.predict(exp1.values))
            Ws = []
            for j in range(0,k):
                temp_j = temp[temp['cluster']==j]
                if temp_j.shape[0]>1:
                    var_j = np.cov(temp_j.drop('cluster',axis=1),rowvar=False,bias=True)
                    Ws.append(temp_j.shape[0]*var_j)
            #teste silhueta
            indices = exp1.sample(frac=0.15).index
            sil_coeff.append(silhouette_score(exp1.iloc[indices], temp['cluster'].iloc[indices], metric='euclidean'))
            #print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff[len(sil_coeff)-1]))
            #outro teste
            W=np.sum(Ws,axis=0)
            W = (W + W.transpose())/2
            SS_T=temp.shape[0]*np.cov(temp.drop('cluster',axis=1),rowvar=False,bias=True)
            SS_T = (SS_T + SS_T.transpose())/2
            B = SS_T-W
            score = np.linalg.det(W)/np.linalg.det(SS_T)
            distortions.append(score)
            #Método Cotovelo Normal
            dists.append(sum(np.min(cdist(exp1, kmeans_teste.cluster_centers_, 'euclidean'), axis=1)) / exp1.shape[0])
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Método Within')
        plt.title('Método cotovelo para k ótimo')
        plt.show()
        #
        plt.plot(K, dists, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Método erro distância euclidiana mínima')
        plt.title('Método cotovelo para k ótimo')
        plt.show()
        #
        plt.plot(K, sil_coeff, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Método Silhueta')
        plt.title('Método cotovelo para k ótimo')
        plt.show()
        #FIM DE CLUSTERIZACAO
        #ENCONTRANDO K-ÓTIMO
        #FIM DE CLUSTERIZACAO
        #ENCONTRANDO K-ÓTIMO
        print('Gerando agrupamento final:')
        k_otimo = 0
        k_otimo = int(input("Escolha o número de k's para agrupamento: "))
        #
        print('Agrupando as variáveis:')
        kmeans = KMeans(n_clusters=k_otimo,random_state=np.random.randint(1, 1000 + 1),n_init=20).fit(exp1)
        #Salvando Kmeans
        self.kmeans = kmeans
        temp = exp1.copy()
        temp['cluster'] = pd.DataFrame(kmeans.predict(exp1.values))
        #
        #Criando modelo de árvore
        print('Criando Modelo de Árvore')
        self.__generate_tree_model(x_samples,temp['cluster'])
        #
        #desnormalizando os cluster centers
        clusters_kmeans = pd.DataFrame(kmeans.cluster_centers_,columns=exp1.columns)
        #desnormalizando nossas variáveis
        for col in clusters_kmeans.columns:
            if col != 'Prob_1':
                clusters_kmeans[col]=norm[col][0]+clusters_kmeans[col]*(norm[col][1]-norm[col][0])
        ####
        ####
        ####
        #Trabalhando 
        print('Colocando todas as análises em uma planilha')
        writer = pd.ExcelWriter(nome_arquivo+'.xlsx', engine='xlsxwriter')
        bold = writer.book.add_format({'bold':True})
        analise_grupos = pd.concat([pd.Series(np.arange(2,20),name='K'),pd.Series(distortions,name='Within'),pd.Series(dists,name='Euclidiana'),pd.Series(sil_coeff,name='Silhueta')],axis=1)
        analise_grupos.to_excel(writer,'Análise_Grupos')
        worksheet = writer.sheets['Análise_Grupos']
        row = 0
        fig = plt.figure(figsize=(4,4))
        plt.text(0.5,0.8,'Qtd de grupos escolhidos:',horizontalalignment="center",color="black",fontsize=20)
        plt.text(0.5,0.3,k_otimo,horizontalalignment="center",color="black",fontsize=140)
        plt.axis('off')
        imgdata = BytesIO()
        fig.savefig(imgdata, format="png")
        imgdata.seek(0)
        worksheet.insert_image(analise_grupos.shape[0]+1, 0, "", {'image_data': imgdata})
        plt.close()
        #imprimindo plots#
        fig = plt.figure(figsize=(5,5))
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Método Within')
        plt.title('Método cotovelo para k ótimo')
        imgdata = BytesIO()
        fig.savefig(imgdata, format="png")
        imgdata.seek(0)
        worksheet.insert_image(0, 6, "", {'image_data': imgdata})
        plt.close()
        #
        fig = plt.figure(figsize=(5,5))
        plt.plot(K, dists, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Método erro distância euclidiana mínima')
        plt.title('Método cotovelo para k ótimo')
        imgdata = BytesIO()
        fig.savefig(imgdata, format="png")
        imgdata.seek(0)
        worksheet.insert_image(0, 14, "", {'image_data': imgdata})
        plt.close()
        #
        fig = plt.figure(figsize=(5,5))
        plt.plot(K, sil_coeff, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Método Silhueta')
        plt.title('Método cotovelo para k ótimo')
        imgdata = BytesIO()
        fig.savefig(imgdata, format="png")
        imgdata.seek(0)
        worksheet.insert_image(0, 22, "", {'image_data': imgdata})
        plt.close()
        #
        centers = np.nan_to_num(kmeans.cluster_centers_)
        distancias = np.zeros((centers.shape[0],centers.shape[0]))
        for center_i in range(0,centers.shape[0]):
            for center_j in range(0,centers.shape[0]):
                distancias[center_i][center_j] = distance.euclidean(centers[center_i], centers[center_j])
        distancias = pd.DataFrame(distancias)
        fig = plt.figure(figsize=(7,7))
        plt.title('Diferença de distância Euclidiana entre os grupos')
        sns.heatmap(distancias, annot=True,cbar=False)
        imgdata = BytesIO()
        fig.savefig(imgdata, format="png")
        imgdata.seek(0)
        worksheet.insert_image(24, 6, "", {'image_data': imgdata})
        plt.close()
        temp.fillna(0,inplace=True)
        self.temp = temp
        #Tratando o problema abaixo:
        temp = pd.concat([temp,probs['Prob_1']],axis=1)
        #colocando os clusters na média
        for cluster in range(0,k_otimo):
            pd.DataFrame(clusters_kmeans.iloc[cluster]).transpose().to_excel(writer,"Grupo_"+str(cluster))
            #
            worksheet = writer.sheets["Grupo_"+str(cluster)]
            #
            worksheet.write_rich_string(3,0,bold,'Média Score')
            worksheet.write(3,1,temp['Prob_1'][temp['cluster'] == cluster].mean())
            #
            worksheet.write_rich_string(4,0,bold,'Desvpad')
            worksheet.write(4,1,str(temp['Prob_1'][temp['cluster'] == cluster].std()))
            #
            worksheet.write_rich_string(5,0,bold,'Score Máx./Mín.')
            worksheet.write(5,1,temp['Prob_1'][temp['cluster'] == cluster].max())
            worksheet.write(5,2,temp['Prob_1'][temp['cluster'] == cluster].min())
            #
            worksheet.write_rich_string(6,0,bold,'Representação Pop.')
            worksheet.write(6,1,temp['Prob_1'][temp['cluster'] == cluster].count()/temp.shape[0])
            #EXPLICACAO MÉDIA DO CLUSTER
            worksheet.write_rich_string(8,0,bold,'Explicação Escore Por Variável do Grupo')
            linha = 9 #Onde começo a escrever.
            var_indices = clusters_kmeans.iloc[cluster].abs().sort_values(ascending=False).index
            for i,indice in enumerate(clusters_kmeans.iloc[cluster].abs().sort_values(ascending=False)):
                if var_indices[i] != 'Prob_1':
                    if clusters_kmeans[var_indices[i]].iloc[cluster] < 0:
                        worksheet.write_rich_string(linha,0,bold,str(var_indices[i]))
                        worksheet.write(linha,1,'Ajuda a diminuir a probabilidade de saída do modelo.')
                    elif clusters_kmeans[var_indices[i]].iloc[cluster] == 0:
                        worksheet.write_rich_string(linha,0,bold,str(var_indices[i]))
                        worksheet.write(linha,1,'Não ajuda na probabilidade de saída do modelo.')
                    else:
                        worksheet.write_rich_string(linha,0,bold,str(var_indices[i]))
                        worksheet.write(linha,1,'Ajuda a aumentar a probabilidade de saída do modelo.')
                linha += 1
            # GRÁFICO COM EXPLICAÇÃO GLOBAL DO GRUPO DE EXPLICABILIDADE
            fig = fig = plt.figure(figsize=(1+int(len(colunas_exp1)*0.2401+0.8911),1+int(len(colunas_exp1)*0.2401+0.8911)))
            names = list(clusters_kmeans.loc[:, clusters_kmeans.columns != 'Prob_1'].iloc[cluster].abs().sort_values(ascending=False).index)
            vals = list(clusters_kmeans[names].loc[:, clusters_kmeans.columns != 'Prob_1'].iloc[cluster].values)
            names.reverse();vals.reverse()
            colors = ['green' if x > 0 else 'red' for x in vals]
            pos = np.arange(len(vals)) + .5
            plt.barh(pos, vals, align='center', color=colors)
            plt.yticks(pos, names)
            title = 'Explicação local para centróide do Grupo %s' % cluster
            plt.title(title)
            imgdata = BytesIO()
            fig.savefig(imgdata, format="png",bbox_inches = 'tight',transparent=True)
            imgdata.seek(0)
            worksheet.insert_image(11, 3, "", {'image_data': imgdata})
            plt.close()
        #salvandos os dados
        print('Salvando a análise em: '+nome_arquivo+'.xlsx.')
        print('Por favor, salve esta classe em pickle.')
        writer.save()
    
    """
        Criando Modelo Simples para rápida interpretabilidade
    """
    def __generate_tree_model(self,X,y):
        tree = lgb.LGBMClassifier(metric='auc',random_state=42,n_estimators=150,learning_rate=0.01,num_leaves=10)
        tree.fit(X,y,eval_metric='auc',verbose=0,categorical_feature=self.categorical_features)
        self.tree_model_kmeans = tree
    
    """
        Função desenvolvida para caso usuário queira apenas explicar uma amostra.
    """
    def explain_alone(self,value,missing_perc=0.20,n_samples=200):
        """
            Returns: Pandas DataFrame com explicações
        """
        #Começando código de amostragem única
        assert missing_perc <= 1 and missing_perc >= 0, 'Valor missing_perc deve estar entre 0 e 1.'
        print('Initializing Cluster Interpretable Model-agnostic Explanations...')
        #amostrando a base
        value.replace([np.nan,np.NAN,np.NaN],[np.nan,np.nan,np.nan],inplace=True)
        #colocando coluna missing nas amostras
        if self.colunas_missing == []:
            for col in self.missings_counts.index:
                if self.missings_counts.loc[col] >= missing_perc:
                    self.colunas_missing.append(col)
        #trabalhando com análise de cada amostra.
        colsmis = []
        for cols in self.colunas_missing:
            colsmis.append(cols+'_nan')
        colunas_exp1 = self.colunas+colsmis
        #próprio algoritmo trata os missings para mim.
        explanations = self.__return_best_coefficients(value.values,n_samples)
        #colocando tudo num dataframe para facilitar.
        exp1 = pd.DataFrame(explanations,index=colunas_exp1).transpose()
        #plotando gráfico de explicabilidade:
        fig = fig = plt.figure(figsize=(1+int(len(colunas_exp1)*0.2401+0.8911),1+int(len(colunas_exp1)*0.2401+0.8911)))
        names = list(exp1.transpose().abs().sort_values(by=0,ascending=False).index)
        vals = list(exp1[names].values[0])
        names.reverse();vals.reverse()
        colors = ['green' if x > 0 else 'red' for x in vals]
        pos = np.arange(len(vals)) + .5
        plt.barh(pos, vals, align='center', color=colors)
        plt.yticks(pos, names)
        title = 'Explicação local para amostra'
        plt.title(title)
        plt.show()
        display(exp1)
        return exp1
    
    """
        Estimativa rápida de grupo de interpretabilidade.
        Esse método deverá ser utilizado com a classe sava em pickle ou até mesmo com toda explicabilidade ainda em memória.
    """
    def estimate_group(self,values):
        """
            Returns: Vector with cluster predictions
        """
        assert self.tree_model_kmeans != None, 'Você deve primeiro, rodar modelo de interpretabilidade completo antes de tentar predizer um grupo.'
        return self.tree_model_kmeans.predict(values)
