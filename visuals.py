from matplotlib.patches import FancyArrowPatch, Circle, ArrowStyle
import numpy as np

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def calculate_node_sizes(graph, net_x_graph):
    def rescale_sizes(importances):
        box_size = len(importances) // 3

        temp_lst = importances.copy()
        temp_lst.sort()

        counter = 0
        cat = 0
        for mmbr in temp_lst:
            index = importances.index(mmbr)

            importances[index] = 900 + cat * cat * cat * 300
            counter += 1

            if counter == box_size:
                cat += 1
                counter = 0

        return importances

    # -------------------
    resulting_sizes = []

    for node in net_x_graph:
        importance = 0
        for tpl in graph:
            if node == tpl[1]:
                importance += tpl[2]
        # resulting_sizes.append(importance*10 + 600)
        resulting_sizes.append(importance)

    return rescale_sizes(resulting_sizes)




graph = [('DEV', 'FUT', 2.0), ('DEV', 'COM', 3), ('DEV', 'ONB', 4), ('DEV', 'DHR', 5), ('DEV', 'EVP', 5), \
         ('DEV', 'RIG', 5.0), ('DEV', 'EWP', 6), ('DEV', 'DNI', 6), ('DEV', 'VEW', 7), ('DEV', 'HPC', 8), \
         ('HPC', 'DNI', 5.0), ('HPC', 'DHR', 5), ('HPC', 'ONB', 5), ('HPC', 'EVP', 4), ('HPC', 'FUT', 2), \
         ('HPC', 'COM', 7.0), ('HPC', 'VEW', 9), ('HPC', 'EWP', 2), ('HPC', 'RIG', 3), ('HPC', 'DEV', 6), \
         ('EWP', 'VEW', 2.0), ('EWP', 'ONB', 3), ('EWP', 'DNI', 3), ('EWP', 'EVP', 2), ('EWP', 'FUT', 8), \
         ('EWP', 'DHR', 2.0), ('EWP', 'COM', 2), ('EWP', 'HPC', 3), ('EWP', 'DEV', 7), ('EWP', 'RIG', 3), \
         ('VEW', 'RIG', 4.0), ('VEW', 'EVP', 8), ('VEW', 'COM', 6), ('VEW', 'ONB', 7), ('VEW', 'DEV', 4),
         ('VEW', 'EWP', 2.0), ('VEW', 'DHR', 5), ('VEW', 'FUT', 7), ('VEW', 'HPC', 6), ('COM', 'ONB', 3), \
         ('COM', 'EVP', 8.0), ('COM', 'HPC', 4), ('COM', 'FUT', 1), ('COM', 'VEW', 8), \
         ('COM', 'DNI', 5.0), ('COM', 'RIG', 2), ('COM', 'DHR', 7), ('COM', 'DEV', 3), ('DNI', 'VEW', 3), \
         ('DNI', 'ONB', 4.0), ('DNI', 'EVP', 6), ('DNI', 'HPC', 6), ('DNI', 'FUT', 2), ('DNI', 'EWP', 3), \
         ('DNI', 'COM', 5.0), ('DNI', 'DEV', 6), ('DNI', 'DHR', 1), ('DNI', 'RIG', 3), ('RIG', 'EWP', 1), \
         ('RIG', 'DNI', 2.0), ('RIG', 'DHR', 3), ('RIG', 'ONB', 3), ('RIG', 'FUT', 4), ('RIG', 'EVP', 6), \
         ('RIG', 'HPC', 4.0), ('RIG', 'VEW', 3), ('RIG', 'COM', 2), ('RIG', 'DEV', 6), ('EVP', 'DNI', 6), \
         ('EVP', 'FUT', 3.0), ('EVP', 'COM', 8), ('EVP', 'RIG', 4), ('EVP', 'DHR', 3), ('EVP', 'EWP', 2), \
         ('EVP', 'DEV', 6), ('EVP', 'HPC', 4), ('EVP', 'VEW', 7), ('ONB', 'DNI', 6), ('ONB', 'VEW', 3), \
         ('ONB', 'RIG', 4), ('ONB', 'DHR', 7), ('ONB', 'COM', 6), ('ONB', 'EWP', 1), ('ONB', 'EVP', 9), \
         ('ONB', 'HPC', 2), ('ONB', 'FUT', 3), ('ONB', 'DEV', 5), ('FUT', 'RIG', 3), ('FUT', 'DNI', 5), \
         ('FUT', 'EVP', 3), ('FUT', 'COM', 4), ('FUT', 'VEW', 4), ('FUT', 'ONB', 4), ('FUT', 'DHR', 1), \
         ('FUT', 'EWP', 3), ('FUT', 'DEV', 2), ('DHR', 'HPC', 5), ('DHR', 'EVP', 7), \
         ('DHR', 'COM', 5), ('DHR', 'DNI', 1), ('DHR', 'RIG', 2), ('DHR', 'VEW', 3), ('DHR', 'ONB', 7), \
         ('DHR', 'FUT', 3), ('DHR', 'DEV', 5)
         ]

graph = [tuple([edo[0], edo[1], float(edo[2])]) for edo in graph]



G = nx.DiGraph()
G.add_weighted_edges_from(graph)


node_sizes = calculate_node_sizes(graph, G)



def draw_network(G, pos, ax, sg=None):
    for n in G:
        c = Circle(pos[n], radius=0.09, alpha=0.1, )
        ax.add_patch(c)
        G.node[n]['patch'] = c
        x, y = pos[n]
    seen = {}
    for (u, v, d) in G.edges(data=True):
        # print('-->',u,v,d)
        n1 = G.node[u]['patch']
        n2 = G.node[v]['patch']
        rad = 0.3
        if (u, v) in seen:
            rad = seen.get((u, v))
            rad = (rad + np.sign(rad) * 0.1) * -1
        alpha = d['weight'] / 10
        color = 'k'

        ast = ArrowStyle("-|>, head_length=" + str(
            (1 + alpha) * (1 + alpha) * (1 + alpha) * (1 + alpha) * (1 + alpha) / 14) + ", head_width=0.2")
        e = FancyArrowPatch(n1.center, n2.center, patchA=n1, patchB=n2,
                            arrowstyle=ast,
                            connectionstyle='arc3,rad=%s' % rad,
                            mutation_scale=17.0,
                            lw=2,
                            alpha=alpha,
                            color=color, linewidth=alpha * alpha + 0.3, antialiased=True)
        seen[(u, v)] = rad
        ax.add_patch(e)
    return e


pos = nx.circular_layout(G)
# pos = nx.spring_layout(G)



plt.figure(4, figsize=(12, 12))

# nx.draw_networkx_edges(G, pos, edge_color=rg, arrows=True, width =0.5, cmap=plt.get_cmap('jet'))


ax = plt.gca()
draw_network(G, pos, ax)
ax.autoscale()

nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'),
                       node_color='green', node_size=node_sizes, alpha=None)
nx.draw_networkx_labels(G, pos)
# nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels)

plt.axis('equal')
# plt.axis('off')


plt.savefig('graph_visual.pdf', dpi=500)

plt.figure(33, figsize=(12, 12))

plt.show()
plt.draw()


