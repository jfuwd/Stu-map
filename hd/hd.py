from flask import Flask,jsonify#导入flask以及处理数据
from flask import render_template,request
import joblib
import numpy as np
from flask_cors import CORS#用来跨域
import json


graph = {'1': [(166, '1', '3'), (103, '1', '2'), (106, '1', '4')],
         '2': [(103, '2', '1'), (74, '2', '3'), (67, '2', '4'), (115, '2', '7'), (100, '2', '9'), (79, '2', '5')],
         '3': [(166, '3', '1'), (74, '3', '2'), (151, '3', '14')],
         '4': [(67, '4', '2'), (106, '4', '1'), (127, '4', '5')],
         '5': [(127, '5', '4'), (79, '5', '2'), (90, '5', '9'), (128, '5', '7')],
         '6': [(73, '6', '14'), (77, '6', '22')],
         '7': [(119, '7', '9'), (115, '7', '2'), (58, '7', '8')],
         '8': [(58, '8', '7'), (102, '8', '14')],
         '9': [(119, '9', '7'), (51, '9', '11'), (58, '9', '10'), (100, '9', '2'), (90, '9', '5')],
         '10': [(58, '10', '9')],
         '11': [(47, '11', '12'), (51, '11', '9')],
         '12': [(47, '12', '11'), (100, '12', '13')],
         '13': [(100, '13', '12'), (27, '13', '16'), (127, '13', '14'), (258, '13', '46')],
         '14': [(102, '14', '8'), (91, '14', '19'), (78, '14', '15'), (127, '14', '13'), (73, '14', '6'),
                (151, '14', '3')],
         '15': [(78, '15', '14'), (94, '15', '19'), (54, '15', '16'), (71, '15', '17')],
         '16': [(27, '16', '13'), (54, '16', '15'), (71, '16', '17'), (79, '16', '18')],
         '17': [(71, '17', '15'), (71, '17', '16'), (79, '17', '18')],
         '18': [(79, '18', '16'), (79, '18', '17')],
         '19': [(91, '19', '14'), (94, '19', '15'), (143, '19', '36'), (42, '19', '20'), (121, '19', '45')],
         '20': [(42, '20', '19')],
         '21': [(60, '21', '22'), (71, '21', '26')],
         '22': [(60, '22', '21'), (70, '22', '24'), (63, '22', '23')],
         '23': [(63, '23', '22')],
         '24': [(70, '24', '22'), (27, '24', '25'), (322, '24', '27')],
         '25': [(27, '25', '24')],
         '26': [(72, '26', '29'), (71, '26', '21')],
         '27': [(322, '27', '24'), (65, '27', '28')],
         '28': [(65, '28', '27')],
         '29': [(72, '29', '26'), (42, '29', '30'), (56, '29', '36')],
         '30': [(42, '30', '29'), (79, '30', '33')],
         '31': [(46, '31', '33'), (1, '31', '32')],
         '32': [(1, '32', '31'), (47, '32', '34')],
         '33': [(46, '33', '31'), (3, '33', '34')],
         '34': [(47, '34', '32'), (3, '34', '33'), (60, '34', '35')],
         '35': [(60, '35', '34'), (113, '35', '57'), (161, '35', '43')],
         '36': [(143, '36', '19'), (56, '36', '29'), (27, '36', '37'), (40, '36', '39'), (85, '36', '45')],
         '37': [(27, '37', '36'), (56, '37', '39'), (3, '37', '38')],
         '38': [(3, '38', '37'), (44, '38', '40'), (5, '38', '41')],
         '39': [(56, '39', '37'), (40, '39', '36'), (2, '39', '40')],
         '40': [(40, '40', '39'), (44, '40', '38'), (4, '40', '41')],
         '41': [(4, '41', '40'), (5, '41', '38')],
         '42': [(4, '42', '43')],
         '43': [(4, '43', '42'), (161, '43', '35'), (153, '43', '56'), (81, '43', '57'), (2, '43', '52')],
         '44': [(61, '44', '47'), (79, '44', '45'), (53, '44', '50')],
         '45': [(79, '45', '44'), (121, '45', '19'), (85, '45', '36')],
         '46': [(258, '46', '13'), (111, '46', '49')],
         '47': [(27, '47', '49'), (61, '47', '44')],
         '48': [(2, '48', '50')],
         '49': [(27, '49', '47'), (81, '49', '54'), (111, '49', '46')],
         '50': [(53, '50', '44'), (53, '50', '52'), (2, '50', '48'), (2, '50', '51')],
         '51': [(2, '51', '50'), (3, '51', '53')],
         '52': [(2, '52', '43'), (3, '52', '55'), (53, '52', '50')],
         '53': [(3, '53', '51')],
         '54': [(136, '54', '56'), (81, '54', '49')],
         '55': [(3, '55', '52')],
         '56': [(136, '56', '54'), (153, '56', '43'), (128, '56', '58')],
         '57': [(71, '57', '58'), (81, '57', '43'), (113, '57', '35'), (1, '57', '59')],
         '58': [(128, '58', '56'), (43, '58', '60'), (71, '58', '57')],
         '59': [(1, '59', '57')],
         '60': [(43, '60', '58'), (2, '60', '61'), (72, '60', '63')],
         '61': [(2, '61', '60'), (2, '61', '62')],
         '62': [(2, '62', '61')],
         '63': [(72, '63', '60')]
         }

def graph2adjacent_matrix(graph):
    vnum = len(graph) + 1
    print(len(graph))
    dict = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    adjacent_matrix = [[0 if row == col else float('inf') for col in range(vnum)] for row in range(vnum)]
    vertices = graph.keys()
    for vertex in vertices:
        for edge in graph[vertex]:
            w, u, v = edge

            adjacent_matrix[int(u)][int(v)] = w
    return adjacent_matrix


def findtrue(endpoints,next):
    truepoint=[]
    plen=len(endpoints)
    print(plen)
    for i in range(0,plen-1):
        j=i+1
        first = endpoints[i]
        truepoint.append(first)
        end = endpoints[j]
        while next[first][end] != end:
            first = next[first][end]
            truepoint.append(first)
    truepoint.append(endpoints[plen - 1])

    print(truepoint)
    return truepoint


def floyd(adjacent_matrix):
    vnum = len(adjacent_matrix)
    a = [[adjacent_matrix[row][col] for col in range(vnum)] for row in range(vnum)]
    nvertex = [[-1 if adjacent_matrix[row][col] == float('inf') else col for col in range(vnum)] for row in range(vnum)]
    # print(adjacent_matrix)
    for k in range(vnum):
        for i in range(vnum):
            for j in range(vnum):
                if a[i][j] > a[i][k] + a[k][j]:
                    a[i][j] = a[i][k] + a[k][j]
                    nvertex[i][j] = nvertex[i][k]
    return nvertex, a


def findway(begin, end, passpoints,a,next):  # int,int,list

    passlen = len(passpoints)
    endpoints = list()
    endpoints.append(begin)

    tp = 0
    for i in range(passlen):
        minlu = 9999
        flag = 0
        for j in range(len(passpoints)):
            if minlu > a[endpoints[tp]][passpoints[j]]:
                minlu = a[endpoints[tp]][passpoints[j]]
                flag = j

        endpoints.append(passpoints[flag])
        passpoints.pop(flag)

    endpoints.append(end)
    truepoint=findtrue(endpoints,next)
    print(endpoints)
    result = ",".join('%s' % id for id in endpoints)
    trueresult=",".join('%s' % id for id in truepoint)
    print(result)
    return result,trueresult






app = Flask(__name__)
CORS(app, supports_credentials=True)

@app.route('/find', methods=['POST'])#路径与模式
def find():
    if request.method == "POST":
        data = request.get_json(silent=True)

        #数据处理
        #取出数据,print(data["pass"] )#取数据,string格式
        begin=int(data["begin"])
        passpoints=data["pass"]
        end=int(data["endpoint"])

        #将passpoints的数据存在list中
        if passpoints!="":
            passarry=passpoints.split(',')
            new_numbers = []

            for n in passarry:
                new_numbers.append(int(n))

            passarry = new_numbers
            print(passarry)

            adjacent_matrix = graph2adjacent_matrix(graph)
            nvertex, a = floyd(adjacent_matrix)
            ### 打印原邻接矩阵 ###
            for i in range(len(adjacent_matrix)):
                for j in range(len(adjacent_matrix[0])):
                    print(adjacent_matrix[i][j], end="\t")
                print()  # 打印一行后换行

            ### 打印经过的顶点 ###
            print()
            for i in range(len(nvertex)):
                for j in range(len(nvertex[0])):
                    print(nvertex[i][j], end="\t")
                print()  # 打印一行后换行
            ### 打印彼此之间的最短距离 ###
            print()
            for i in range(len(a)):
                for j in range(len(a[0])):
                    print(a[i][j], end="\t")
                print()  # 打印一行后换行

            #最短路径函数
            endpasspoint,truepasspoint=findway(int(begin),int(end),passarry,a,nvertex)
        else:
            adjacent_matrix = graph2adjacent_matrix(graph)
            nvertex, a = floyd(adjacent_matrix)
            endpasspoint, truepasspoint = findway(int(begin), int(end), [], a, nvertex)


        json_data = {
            'status': True,
            'enddata': {
                'pass':endpasspoint,
                'truepass':truepasspoint
            }
        }
        return jsonify(json_data)



if __name__ == '__main__':
    app.run(host="127.0.0.1",debug=True)#直接host127.0.0.1在本地
    #app.run(debug=True)