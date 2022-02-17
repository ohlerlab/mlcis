class_index = 0

ind1 = [i[0] for i in sorted(enumerate(pred[:,class_index]), key=lambda x:x[1],reverse=True) if y_test[i[0],class_index]==1 and pred[i[0],class_index] > 0.90]

ex_seq1 = np.array([igres.explain(x_test[i],outc=class_index,reference=False) for i in ind1])

att_val_add=np.zeros((84,100))

for i in range(64):
    att_values=ex_seq1[i].squeeze()
    att_val_add=np.add(att_val_add,att_values)

att_val_add = att_val_add[::-1]

ax=Visualize(att_val_add,0,savefig=False)

ax=Visualize(att_val_add,0,polarity= 'negative',savefig=False)

