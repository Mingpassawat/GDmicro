import re
import logging
import numpy as np
import wandb_logger

LOGGER = logging.getLogger(__name__)

def cal_acc_cv(infile,ofile):
    f=open(infile,'r')
    o=open(ofile,'w+')

    dtrain={} # Fold id -> [train_acc]
    dval={} # Fold id -> [val_acc]

    dtrainu={}
    dvalu={}

    dtrainm={}
    dvalm={}

    dtrainum={}
    dvalum={}

    while True:
        line=f.readline().strip()
        if not line:break
        if not re.search('Fold', line) and not re.search('Epoch',line):continue
        if re.search('Fold',line):
            fid=line.split()[-1]
            dtrain[fid]=[]
            dval[fid]=[]
            dtrainu[fid]=[]
            dvalu[fid]=[]

            dtrainm[fid]=0
            dvalm[fid]=0

            dtrainum[fid]=0
            dvalum[fid]=0
            continue
        ele=line.split()
        dtrain[fid].append(float(ele[5]))
        dval[fid].append(float(ele[9]))
        dtrainu[fid].append(float(ele[13]))
        dvalu[fid].append(float(ele[15]))


        if float(ele[5])>dtrainm[fid]:
            dtrainm[fid]=float(ele[5])
        if float(ele[13])>dtrainum[fid]:
            dtrainum[fid]=float(ele[13])
        if float(ele[9])>dvalm[fid]:
            dvalm[fid]=float(ele[9])
        if float(ele[15])>dvalum[fid]:
            dvalum[fid]=float(ele[15])

    avg_train=[]
    avg_val=[]

    avg_tu=[]
    avg_vu=[]

    bt_train=[]
    bt_val=[]

    bt_trainu=[]
    bt_valu=[]

    for i in dtrain:
        o.write('The best train acc of Fold '+str(i)+' is '+str(dtrainm[i])+'\n')
        o.write('The best train AUC of Fold '+str(i)+' is '+str(dtrainum[i])+'\n')
        o.write('The best val acc of Fold '+str(i)+' is '+str(dvalm[i])+'\n')
        o.write('The best val AUC of Fold '+str(i)+' is '+str(dvalum[i])+'\n')

        LOGGER.info('CV Fold %s | best_train_acc=%.4f best_train_auc=%.4f best_val_acc=%.4f best_val_auc=%.4f',
                str(i), float(dtrainm[i]), float(dtrainum[i]), float(dvalm[i]), float(dvalum[i]))
        wandb_logger.log({
            'cv/fold': int(i),
            'cv/best_train_acc': float(dtrainm[i]),
            'cv/best_train_auc': float(dtrainum[i]),
            'cv/best_val_acc': float(dvalm[i]),
            'cv/best_val_auc': float(dvalum[i]),
        })

        avg_train.append(np.mean(dtrain[i]))
        avg_val.append(np.mean(dval[i]))

        avg_tu.append(np.mean(dtrainu[i]))
        avg_vu.append(np.mean(dvalu[i]))

        bt_train.append(dtrainm[i])
        bt_val.append(dvalm[i])

        bt_trainu.append(dtrainum[i])
        bt_valu.append(dvalum[i])

    o.write('Final: The averaga train acc is '+str(np.mean(bt_train))+'\n')
    o.write('Final: The average train AUC is '+str(np.mean(bt_trainu))+'\n')
    o.write('Final: The average val acc is '+str(np.mean(bt_val))+'\n')
    o.write('Final: The average val AUC is '+str(np.mean(bt_valu))+'\n')

    LOGGER.info('CV Summary | avg_best_train_acc=%.4f avg_best_train_auc=%.4f avg_best_val_acc=%.4f avg_best_val_auc=%.4f',
                float(np.mean(bt_train)), float(np.mean(bt_trainu)), float(np.mean(bt_val)), float(np.mean(bt_valu)))
    wandb_logger.summary_update({
        'cv/avg_best_train_acc': float(np.mean(bt_train)),
        'cv/avg_best_train_auc': float(np.mean(bt_trainu)),
        'cv/avg_best_val_acc': float(np.mean(bt_val)),
        'cv/avg_best_val_auc': float(np.mean(bt_valu)),
    })
            
