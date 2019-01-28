# TTN_pytorch
TextTopicNet implementation in Pytorch 

The filepaths in the python code are messed up, in order to make it work you will have to change the paths to your local ones.
In order to get the Wikipedia dataset and train the LDA model, follow instructions at the original TextTopicNet work at this link: https://github.com/lluisgomez/TextTopicNet

#10K samples: there are two folders corresponding to 10K output vectors of CAFFE and PYTORCH models. Best way to read the contents of these files is simply:
  - CAFFE output: ```nparray = np.asarray(json.load(open('<path_to_file_folder>/topic_probs.json'))['topic_probs'])```
  - PYTORCH output: ```nparray = np.loadtxt('<path_to_file_folder>/outputVector.npy', delimiter=';  ')```
  
