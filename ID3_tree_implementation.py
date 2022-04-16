import concurrent.futures
import ID3_functions as id3f

class Tree():

    #Inicializar a classe com o nome para ser um identificador e a list_dict que funcionara como método de decisão
    def __init__(self,name = None):
        self.name = name
        self.tree = {}
        self.first_category = 0

    #Função auxiliar para percorrer a arvore e chegar ate o nó definido pela lista
    def __get_node(self,path,tree):

        #Selecionar o primeiro flho do dicionario
        node = tree.get(path[0])

        #Selecionar os demais filhos que constam em list
        for _ in range(len(path)-1):
            node = node.get(path[_+1])
    
        return node

    #Definir a função que a partir de um atributo decide qual o proximo que deve ser avaliado
    def __chose_next_node(self,list_categories,path,df,tree):

        #Escolher um dataframe que contenha só as linhas que correspondem a categoria indicada em path
        for attribute,df_principal in df.groupby(path[-1]):

            #Definir entropia inicial desse nó:
            inicial_entropy = df_principal.Rating.value_counts(normalize = True).map(lambda x: -x*id3f.log(x)).sum()

            #Definir a entropia para cada categoria disponivel
            entropy_gain = {}
            for category in (set(list_categories) - set(path)):
                total = 0
                soma = 0

                for p,df_subselected in df_principal.groupby(category):
                    attribute_column = df_subselected.Rating
                    total += attribute_column.count()
                    soma += (attribute_column.value_counts(normalize = True).map(lambda x: -x*id3f.log(x)).sum())*attribute_column.count()
                entropy_gain[category] = inicial_entropy - soma/total

            #Selecionar o maior ganho de entropia
            category,value = sorted(entropy_gain.items(), key = lambda item: item[1], reverse = True)[0]

            #Caso exista ganho de entropia
            if value > 0.01:

                #Adicionar o valor 
                self.__get_node(path,tree)[attribute] = {}
                path.append(attribute)

                #Adicionar categoria escolhida
                self.__get_node(path,tree)[category] = {}
                path.append(category)

                #escolher proxima categoria
                self.__chose_next_node(list_categories,path,df_principal,tree)

                #Retirar os parametros de profundidade do nó
                path.pop()
                path.pop()

            #Caso não exista ganho de entropia, escolher a nota que sera dada
            else:
                self.__get_node(path,tree)[attribute] = df_principal.Rating.value_counts().index[0]
                

    #Função para definir a arvore
    def define_tree(self,list_categories,path,df_aux):
        tree = {}
        tree.update({self.first_category: {}})
        self.__chose_next_node(list_categories,path,df_aux,tree)
        return tree

    #Treinar o modelo a partir do dataframe, é necessario que a primeira coluna contenha o atribito que se deseja prever
    def train(self,df):

        #Definir as cateorias existentes
        path = []
        list_categories = df.columns[1::]

        #Definir a entropia inicial do dataset
        inicial_entropy = df.Rating.value_counts(normalize = True).map(lambda x: -x*id3f.log(x)).sum()

        #Definir a entropia de cada categoria para o primeiro nó
        entropy_gain = {}
        for attribute in list_categories:
            total = 0
            soma = 0
            for _,line in df.groupby(attribute):
                attribute_column =line.Rating
                total += attribute_column.count()
                soma += (attribute_column.value_counts(normalize = True).map(lambda x: -x*id3f.log(x)).sum())*attribute_column.count()
            entropy_gain[attribute] = inicial_entropy - soma/total


        #Selecionar o primeiro atributo que sera analisado
        category,_ = sorted(entropy_gain.items(), key = lambda item: item[1], reverse = True)[0]

        #Adicionar elemento a arvore e a variavel path
        self.first_category = category
        path.append(category)
        self.tree.update({category: {}})

        #Dividir o dataset entre os executors e as variaveis
        #Inicializar variaveis
        list_df = []
        replicate_path = []
        replicate_list_categories = []
        list_trees = []

        #Criar vetores para aplicar a paralelização de processos
        for _,line in df.groupby(self.first_category):
            list_df.append(line)
            replicate_path.append(path)
            replicate_list_categories.append(list_categories)
            aux = {}
            list_trees.append(aux)

        #Selecionar os proximos nós utilizando paralelização
        with concurrent.futures.ProcessPoolExecutor(5) as executor:

            #Executar os processos em paralelo
            results = executor.map(self.define_tree,replicate_list_categories,replicate_path,list_df)

            #Concatenar os resultados obtidos
            auxiliar_tree = {}
            for result in results:
                auxiliar_tree.update(result.get(self.first_category))
            
            #Transferir a arvore treinada para a principal
            self.tree.get(self.first_category).update(auxiliar_tree)

    def get_tree(self):
        return self.tree
