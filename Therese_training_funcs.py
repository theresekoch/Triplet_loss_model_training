import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import torchvision.datasets as datasets
import os
import torch
from itertools import combinations

from utils import TripletSelector, pdist





def weighted_semihard_negative(loss_values, margin, prop_semihard):
    #randomly determine whether a semihard negative or a hard negative should be returned
    if np.random.random() < prop_semihard: 
        semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
        return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None
    else:
        hard_negatives = np.where(loss_values > margin)[0]
        return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

class MetricThresholdMargin():
    #class to implement a dynamic margin parameter which updates based on a given metric crossing a threshold value. 
    def __init__(self, min_margin, max_margin, margin_step, metric_threshold):
        self.min_margin = min_margin
        self.max_margin = max_margin
        #self.n_epochs = n_epochs
        self.margin_step = margin_step
        self.current_margin = min_margin
        self.metric_threshold = metric_threshold
    
    def update_margin(self, metric):
        #check if metric is below threshold
        if metric < self.metric_threshold: 
            #check if margin can be updated without exceeding max
            if self.current_margin + self.margin_step <= self.max_margin:
                self.current_margin += self.margin_step
            else : 
                self.current_margin = self.max_margin
            #print(self.current_margin)
        return self.current_margin
    
    def get_current_margin(self):
        return self.current_margin

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

class CustomDatasetFolder(datasets.DatasetFolder):
    ## creates dataset of spectrograms from directory of Bird_ID folders which each have subfolders for each syllable type. 
    ## Also creates dict mapping class numbers to bird ID so that sibling negatives can be excluded. 
    def __init__(self, 
                root: str,
                loader: Callable[[str], Any],
                extensions: Optional[Tuple[str, ...]] = None,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                is_valid_file: Optional[Callable[[str], bool]] = None, 
                Bird_ID: Optional[str] = None, 
                train: Optional[bool] = None, 
                exclude: Optional[list] = None):

        self.train = train
        self.Bird_ID = Bird_ID
        self.exclude = exclude

        if self.train is not None:
            if self.Bird_ID is None:
                raise ValueError("If a train value is provided, a Bird_ID must also be provided")

        if self.train is None: 
            if self.Bird_ID is not None: 
                raise ValueError("For train/test splitting a value for `train` must be provided in addition to `Bird_ID`.")

        super().__init__(root, 
                        loader = loader, 
                        extensions = extensions, 
                        transform = transform, 
                        target_transform = target_transform, 
                        is_valid_file = is_valid_file
                        )

        self.idx_to_class ={idx : class_label for class_label, idx in self.class_to_idx.items()}#########added as a test
        
    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.

        See :class:`DatasetFolder` for details.
        """

        #if train = true return dataset with syllables from all birds except Bird_ID and "exclude" birds
        if self.train is True:
            classes = []
            
            for subdirectory in os.scandir(directory):
                if subdirectory.name == self.Bird_ID:
                    continue
                if subdirectory.name in self.exclude: 
                    continue
                for sub_subdirectory in os.scandir(subdirectory):
                    if sub_subdirectory.is_dir():
                        classes.append(directory + subdirectory.name + "/" + sub_subdirectory.name)

        if self.train is False:
            classes = []

            subdirectory = directory + self.Bird_ID

            for sub_subdirectory in os.scandir(subdirectory):
                if sub_subdirectory.is_dir():
                    classes.append(subdirectory + "/" + sub_subdirectory.name)


        classes = sorted(classes)

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

def get_Bird_ID_from_label(label, idx_to_class):
    ## Returns ID of bird producing syllable with a given int label from the custom dataset loader. 
    ## Necessary for excluding sibling negative triplets. 
    class_label = idx_to_class[label]
    Bird_ID = class_label.split("/")[-2]
    return Bird_ID

def get_siblings(Bird_ID, tutor_dict): 
    #returns a list of the bird IDs of all birds that share a tutor with the input bird. 
    tutor_ID = tutor_dict[Bird_ID]
    #an isolate shouldn't be considered the sibling of other isolates
    if tutor_ID == "Isolate":
        return []
        
    siblings = []
    for pupil, tutor in tutor_dict.items():
        if tutor == tutor_ID:
            if pupil != Bird_ID:
                siblings.append(pupil)
    return siblings

def get_labels_from_Bird_ID(bird_IDs, idx_to_class):
    ## returns the int labels for all classes produced by birds in the list of bird_IDs. 
    ## This is necessary for excluding sibling negative triplets. 
    labels = []
    for idx, class_label in idx_to_class.items():
        for Bird_ID in bird_IDs:
            if Bird_ID in class_label:
                labels.append(idx)
    return labels


class NoSibsPropSemihardTripletSelector(TripletSelector):
    """
    Returns prop_semihard proportion of semihard triplets, and 1 - prop_semihard proportion of true hard triplets
    Where none of the triplets contain a negative from a bird that shares a tutor with the anchor syllable bird. 
    This is done to avoid training with triplets containing different versions of the same syllable, learned by 
    copying the same tutor as this would bias to model towards learning vocalizer differences vs. syllable differences. 
    """
    def __init__(self, margin, prop_semihard, idx_to_class, pup_tut_dict, cpu = True):
        super(NoSibsPropSemihardTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.prop_semihard = prop_semihard
        self.idx_to_class = idx_to_class
        self.pup_tut_dict = pup_tut_dict

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()

        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            
            #create mask to exclude siblings
            curr_bird = get_Bird_ID_from_label(label, self.idx_to_class)
            siblings = get_siblings(curr_bird, self.pup_tut_dict)
            if len(siblings) > 0:
                sibling_labels = get_labels_from_Bird_ID(siblings, self.idx_to_class)
            else:
                sibling_labels = []

            sibling_labels_mask = np.isin(labels, sibling_labels)

            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask) & np.logical_not(sibling_labels_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:,0], anchor_positives[:,1]]

            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                selected_negative = weighted_semihard_negative(loss_values, margin = self.margin, prop_semihard = self.prop_semihard)
                if selected_negative is not None:
                    selected_negative = negative_indices[selected_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], selected_negative])
            


        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])
            
        triplets = np.array(triplets)

        return torch.LongTensor(triplets)