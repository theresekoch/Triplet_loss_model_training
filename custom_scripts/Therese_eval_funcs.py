import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import torchvision.datasets as datasets
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture




class CustomDatasetFolderNoClass(datasets.DatasetFolder):
    """creates dataset compatible with embedding net evaluation functions for unlabeled syllable spectrograms. 
    
    """
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
        classes = [self.Bird_ID]
        

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


def extract_embeddings(dataloader, model, n_dim, cuda = True):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), n_dim))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels

def single_bird_extract_embeddings(Bird_ID, all_birds_dir, model, n_dim = 2, cuda = True):
    
    single_bird_dataset = CustomDatasetFolderNoClass(all_birds_dir, extensions = (".npy"), loader = np.load, 
                                    train = False, Bird_ID = Bird_ID)

    single_bird_dataloader = torch.utils.data.DataLoader(single_bird_dataset, batch_size=64, shuffle=False)

    single_bird_embeddings, single_bird_labels = extract_embeddings(single_bird_dataloader, model, n_dim = n_dim, cuda = cuda)

    return single_bird_embeddings, single_bird_labels


def sample_embedding_rows(embedding_array, num_samples):
    if embedding_array.shape[0] > num_samples:
        return embedding_array[np.random.choice(embedding_array.shape[0], num_samples, replace = False)]

    else: 
        return embedding_array[np.random.choice(embedding_array.shape[0], num_samples, replace = True)]


def calc_Dkl(P_embedding, P_n_components, Q_embedding, Q_n_components, prop_split = 0.75):
    #split embeddings to withold data from GMM fit
    P_embedding_train, P_embedding_test = train_test_split(P_embedding, train_size = prop_split)
    Q_embedding_train, Q_embedding_test = train_test_split(Q_embedding, train_size = prop_split)

    #get the number of syllable types in P and Q
    P_n_sylls = P_n_components
    Q_n_sylls = Q_n_components

    #fit GMM to P and Q train sets
    P_GMM = GaussianMixture(n_components= P_n_sylls).fit(P_embedding_train)
    Q_GMM = GaussianMixture(n_components = Q_n_sylls).fit(Q_embedding_train)

    #calculate likelihood of withheld P data in P_GMM and Q_GMM
    p_P_likelihood = P_GMM.score(P_embedding_test)
    p_Q_likelihood = Q_GMM.score(P_embedding_test)

    #calculate Dkl estimate
    D_kl = (np.log2(np.e) * (p_P_likelihood - p_Q_likelihood)) / len(P_embedding)

    return D_kl
