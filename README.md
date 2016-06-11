# Triplet loss for Caffe

Introduce triplet loss layer to caffe.<br>
Concretely, we use cosine matric to constrain the distance between samples among same label/different labels.

## Useage
1st, you need create a ordered file list for training. <br>
This file list control the exactly data read-in order during training phase. Suppose in each mini-batch, you have data from 4 labels and 2 samples in each label, then the content of the file list should be like this:<br>
<pre><code>
  img_path_1_from_label_1 label_1
  img_path_2_from_label_1 label_1
  img_path_1_from_label_2 label_2
  img_path_2_from_label_2 label_2
  ...
  img_path_1_from_label_4 label_4
  img_path_2_from_label_4 label_4
</code></pre>

2nd, define network structure in your train_val.prototxt.<br>
Setup SampleTripletLayer to sample triplets in each mini-batch. Currently, triplets are made up by all anchor-positive pairs in the sample label and one hardest negative sample from other labels.<br>
<pre><code>
  layer {
    name: "sample_triplet"
    type: "SampleTriplet"
    bottom: "fully_connected_feature"
    top: "triplet"
    sample_triplet_param {
      label_num: 4
      sample_num: 2
    }
  }
</code></pre>
Setup TripletLossLayer to calculate loss.<br>
<pre><code>
  layer {
    name: "triplet_loss"
    type: "TripletLoss"
    bottom: "fully_connected_feature"
    bottom: "triplet"
    top: "triplet_loss"
    triplet_loss_param {
      margin: 0.1
    }
    loss_weight: 1
  }
</code></pre>
