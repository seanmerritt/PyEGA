import pandas as pd
from sklearn.covariance import GraphicalLasso, shrunk_covariance
import igraph
from igraph import Graph
import numpy as np
from sklearn.preprocessing import StandardScaler
from cdlib import algorithms

class EGA():
    def __init__(self):
        print("Getting ready for EGA")
        self.cvm = None
        self.glasso_cvm = None
        self.graph = None
        
    def df_to_matrix(self, data):
        scaler = StandardScaler()
        self.cvm = np.cov(scaler.fit_transform(data.T))
    
    def to_glasso_cvm(self):
        alphaRange = 10.0 ** np.arange(-8,0) # 1e-7 to 1e-1 by order of magnitude
        matrix = shrunk_covariance(self.cvm,.8)
        glasso_matrix = GraphicalLasso(max_iter = 1000).fit(matrix, 0.1)
        self.glasso_cvm =  pd.DataFrame(glasso_matrix.covariance_)
        '''
        for alpha in alphaRange:
            try: 
                self.glasso_cvm = np.array(GraphicalLasso(max_iter = 1000).fit(matrix, alpha))
                print("Calculated graph-lasso covariance matrix for alpha=%s"%alpha)
            except FloatingPointError:
                print("Failed at alpha=%s"%alpha)
        '''
    def cvm_to_graph(self):
        A = self.glasso_cvm.values

        # Create graph, A.astype(bool).tolist() or (A / A).tolist() can also be used.
        self.graph = igraph.Graph.Adjacency((A > 0).tolist())

        # Add edge weights and node labels.
        self.graph.es['weight'] = A[A.nonzero()]
        self.graph.vs['label'] =  self.glasso_cvm.index #/a.columns
        
    
    def detect_communities(self):
        print(self.graph.community_walktrap().as_clustering().membership)
    
    def fit(self, data):
        self.df_to_matrix(data)
        self.to_glasso_cvm()
        self.cvm_to_graph()
        igraph.plot(self.graph)
        #self.detect_communities()