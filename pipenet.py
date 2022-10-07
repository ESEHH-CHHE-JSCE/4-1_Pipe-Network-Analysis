# -*- coding: Shift-JIS -*

import numpy as np
from numpy.linalg import solve
from pandas import Series,DataFrame
import pandas as pd
import networkx as nx


###Rの算出###
df3 = pd.read_csv('data/03_pipe.csv')
df3['R'] = 0.27853 * df3['C'] * df3['D']**2.63 * df3['L']**(-0.54)

Count=0

###Sijの算出###
df2=pd.read_csv('data/02_junction.csv')
Ans=np.ones(len(df2)-1)

###管網図の読み込み（https://www.yutaka-note.com/entry/networkx#%E5%9F%BA%E6%9C%AC%E7%94%A8%E8%AA%9E%E3%81%AE%E6%95%B4%E7%90%86）
# Graphオブジェクトの作成
G = nx.Graph()

# nodeデータの追加
G.add_nodes_from(df2["i"].to_numpy().tolist())

#edge(辺)データの追加
F=df3[["i","j"]].to_numpy().tolist()
G.add_edges_from(F,length=df3["L"])

# 隣接行列(B)の出力
B=nx.to_numpy_matrix(G)
B=np.array(B)

while max(abs(Ans))>1e-04:
    ret = pd.merge(df3, df2, left_on="i", right_on="i",how="left") #Eiを"E'_x"として記述
    df3 = ret
    ret = pd.merge(df3, df2, left_on="j", right_on="i",how="left") #Ejを"E'_y"として記述
    df3 = ret
    df3=df3.drop(["demand_x","i_y","demand_y"],axis=1)
    df3=df3.rename(columns={"i_x": "i","E_x":"Ei","E_y":"Ej"})
    df3["Sij"] = df3["R"] * abs(df3["Ei"] - df3["Ej"]) ** (0.54-1)
    
    
    
    ###左辺の負の係数(L2)を算出
    Sij=df3[["i","j","Sij"]].values  #df3よりSijを抽出
    L2 = np.zeros((len(df2),len(df2)))
    for m in range(len(Sij)):
        i,j,s=Sij[m]
        L2[int(i)-1][int(j)-1]=s
        L2[int(j)-1][int(i)-1]=s
    
    ###左辺の正の係数を算出(L1)
    L1=np.sum(L2, axis=1)
    L1=np.diag(L1)
    
    #左辺係数(L=L1-L2)の算出###
    L=L1-L2
    
    ###右辺(R)の算出###
    ###(Ei-Ej)を”R4”として算出
    e=df2[["i","E"]] .values
    R1 = np.zeros((len(df2),len(df2)))
    for m in range(len(e)):
        i,E=e[m]
        R1[m]=E
    
    R2=R1.T    
    R3=R1-R2
    R4=R3*B
    
    #Sij(Ei-Ej)を”R5”として算出
    R5=R4*L2
    #print(R5)
    #右辺（Marlow）を"R"として算出
    R6=np.sum(R5, axis=1) #R5の行を合計
    q=df2["demand"].values
    R=-1/0.54*(R6+q)
    
    ###連立方程式を解く
    df1 = pd.read_csv('data/01_tank.csv')
    q=df1["i"] .values-1
    #左辺係数のtankに該当するi行およびj列を削除
    L=np.delete(L, q, 0)
    L=np.delete(L, q, 1)
    
    #右辺係数のtankに該当するi行を削除
    R=np.delete(R, q, 0)

    Ans=solve(L, R)
    
    ####各節点エネルギーに解を足しこむ
    Ans2=np.insert(Ans,q,0)
    df2["E"]=df2["E"]+Ans2

    ####df3の体裁を整える
    df3=df3.drop(["Ei","Ej","Sij"],axis=1)

    Count=Count+1

print("計算回数=",Count,"回")
print(df2)



######以下、参考
# ネットワークの可視化
#nx.draw(G, with_labels = True)
#plt.show()
