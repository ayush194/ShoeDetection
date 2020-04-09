import keras.losses as losses
import tensorflow.multiply as tf.multiply

def myLoss(ytrue,ypred):
    true_box_prob = ytrue[:,:2]
    true_box_coords1 = ytrue[:,2:6]
    true_box_coords2 = ytrue[:,6:10]
    pred_box_prob = ypred[:,:2]
    pred_box_coords1 = ypred[:,2:6]
    pred_box_coords2 = ypred[:,6:10]
    r1= losses.mse(y_true=true_box_coords1,y_pred=pred_box_coords1)
    r2= losses.mse(y_true=true_box_coords2,y_pred=pred_box_coords2)
    r1 = tf.multiply(r1 ,true_box_prob[:,0])
    r2 = tf.multiply(r2 ,true_box_prob[:,1])
    classification_loss = losses.binary_crossentropy(y_true=true_box_prob,y_pred=pred_box_prob)
    return (r1+r2) + classification_loss

def myLoss2(ytrue,ypred):
    true_box_prob = ytrue[:,:2]
    true_box_coords1 = ytrue[:,2:52]
    true_box_coords2 = ytrue[:,52:102]
    pred_box_prob = ypred[:,:2]
    pred_box_coords1 = ypred[:,2:52]
    pred_box_coords2 = ypred[:,52:102]
    r1= losses.mse(y_true=true_box_coords1,y_pred=pred_box_coords1)
    r2= losses.mse(y_true=true_box_coords2,y_pred=pred_box_coords2)
    r1 = tf.multiply(r1 ,true_box_prob[:,0])
    r2 = tf.multiply(r2 ,true_box_prob[:,1])
    classification_loss = losses.binary_crossentropy(y_true=true_box_prob,y_pred=pred_box_prob)
    return (r1+r2) + classification_loss