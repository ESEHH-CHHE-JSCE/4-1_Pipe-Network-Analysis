# -*- coding: Shift-JIS -*

import numpy as np
from numpy.linalg import solve
from pandas import Series,DataFrame
import pandas as pd
import networkx as nx


###R�̎Z�o###
df3 = pd.read_csv('data/03_pipe.csv')
df3['R'] = 0.27853 * df3['C'] * df3['D']**2.63 * df3['L']**(-0.54)

Count=0

###Sij�̎Z�o###
df2=pd.read_csv('data/02_junction.csv')
Ans=np.ones(len(df2)-1)

###�ǖԐ}�̓ǂݍ��݁ihttps://www.yutaka-note.com/entry/networkx#%E5%9F%BA%E6%9C%AC%E7%94%A8%E8%AA%9E%E3%81%AE%E6%95%B4%E7%90%86�j
# Graph�I�u�W�F�N�g�̍쐬
G = nx.Graph()

# node�f�[�^�̒ǉ�
G.add_nodes_from(df2["i"].to_numpy().tolist())

#edge(��)�f�[�^�̒ǉ�
F=df3[["i","j"]].to_numpy().tolist()
G.add_edges_from(F,length=df3["L"])

# �אڍs��(B)�̏o��
B=nx.to_numpy_matrix(G)
B=np.array(B)

while max(abs(Ans))>1e-04:
    ret = pd.merge(df3, df2, left_on="i", right_on="i",how="left") #Ei��"E'_x"�Ƃ��ċL�q
    df3 = ret
    ret = pd.merge(df3, df2, left_on="j", right_on="i",how="left") #Ej��"E'_y"�Ƃ��ċL�q
    df3 = ret
    df3=df3.drop(["demand_x","i_y","demand_y"],axis=1)
    df3=df3.rename(columns={"i_x": "i","E_x":"Ei","E_y":"Ej"})
    df3["Sij"] = df3["R"] * abs(df3["Ei"] - df3["Ej"]) ** (0.54-1)
    
    
    
    ###���ӂ̕��̌W��(L2)���Z�o
    Sij=df3[["i","j","Sij"]].values  #df3���Sij�𒊏o
    L2 = np.zeros((len(df2),len(df2)))
    for m in range(len(Sij)):
        i,j,s=Sij[m]
        L2[int(i)-1][int(j)-1]=s
        L2[int(j)-1][int(i)-1]=s
    
    ###���ӂ̐��̌W�����Z�o(L1)
    L1=np.sum(L2, axis=1)
    L1=np.diag(L1)
    
    #���ӌW��(L=L1-L2)�̎Z�o###
    L=L1-L2
    
    ###�E��(R)�̎Z�o###
    ###(Ei-Ej)���hR4�h�Ƃ��ĎZ�o
    e=df2[["i","E"]] .values
    R1 = np.zeros((len(df2),len(df2)))
    for m in range(len(e)):
        i,E=e[m]
        R1[m]=E
    
    R2=R1.T    
    R3=R1-R2
    R4=R3*B
    
    #Sij(Ei-Ej)���hR5�h�Ƃ��ĎZ�o
    R5=R4*L2
    #print(R5)
    #�E�ӁiMarlow�j��"R"�Ƃ��ĎZ�o
    R6=np.sum(R5, axis=1) #R5�̍s�����v
    q=df2["demand"].values
    R=-1/0.54*(R6+q)
    
    ###�A��������������
    df1 = pd.read_csv('data/01_tank.csv')
    q=df1["i"] .values-1
    #���ӌW����tank�ɊY������i�s�����j����폜
    L=np.delete(L, q, 0)
    L=np.delete(L, q, 1)
    
    #�E�ӌW����tank�ɊY������i�s���폜
    R=np.delete(R, q, 0)

    Ans=solve(L, R)
    
    ####�e�ߓ_�G�l���M�[�ɉ��𑫂�����
    Ans2=np.insert(Ans,q,0)
    df2["E"]=df2["E"]+Ans2

    ####df3�̑̍ق𐮂���
    df3=df3.drop(["Ei","Ej","Sij"],axis=1)

    Count=Count+1

print("�v�Z��=",Count,"��")
print(df2)



######�ȉ��A�Q�l
# �l�b�g���[�N�̉���
#nx.draw(G, with_labels = True)
#plt.show()
