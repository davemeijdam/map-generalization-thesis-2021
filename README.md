# map-generalization-thesis-2021
The problem is that the current situation of projecting objects, such as polygons and lines, on a map is hurting the performance of a GIS. The present systems only decide wether an object gets shown or not, without any intermediate steps. The problem can be solved by applying ‘the best’ simplification algorithm to an object to decrease the number of data points(coordinates) within an object. The best simplification algorithm will be determined by the data size reduction, the process time of the algorithm and the retainment of the visual representation of an object. A machine learning model will be develop and trained to decide which simplification algorithm fits an individual object the best. The machine learning model will be implemented in (a part of) the GIS and will be tested on performance with the present GIS.


# TODO
https://github.com/pytorch/

0. Send an email to the thesis coordinator. Ask him about GPUs which you can use. Ask him whether you are allowed to use SURF sara. https://userinfo.surfsara.nl/. If the answer is YES, then read how to create an account on Lisa SURF sara. and specify that you need access to GPUs. Add me as a coordinator of the project.
    - to check whether pytorch uses GPUs do the following
    ```python
    import torch
    torch.cuda.is_available()
    ```
1. [May be you can do it while you are waiting for the answer] Achieve at least 99.2 on MNIST. Do everything in a notebook.
    - take a look at at least 1 tutorial.
    - take a look at https://github.com/pytorch/examples/blob/master/mnist/main.py
2. Create a dataset for the provided data.
Use `utils/datasets.py` for this. create the file and add a content to it. 
    - Read about pytorch custom datasets. Take a look at `torchvision/datasets/mnist.py`
3. Create a loader for the dataset. add it to `utils/loaders.py`
What's loader? `torch.utils.data.DataLoader`
4. [It's optional] Read about Graph Neural Networks (GNNs) 
https://tkipf.github.io/graph-convolutional-networks/