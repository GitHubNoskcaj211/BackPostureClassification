import torch

class ImageCategoricalPredictor(torch.nn.Module):
    def __init__(self, image_shape, output_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 20, 5, stride=1)
        self.conv2 = torch.nn.Conv2d(20, 40, 5, stride=1)
        self.conv3 = torch.nn.Conv2d(40, 60, 5, stride=1)
        flattened_dim = 2 * (image_shape[0]-4-2-4-2-4-2) * (image_shape[1]-4-2-4-2-4-2) * 60
        self.linear1 = torch.nn.Linear(flattened_dim, 100)
        self.linear2 = torch.nn.Linear(100, 2)
        self.max_pool = torch.nn.MaxPool2d(3, stride=1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        image1 = x[:,:3]
        image2 = x[:,3:]

        def convolve(image):
            image = self.conv1(image)
            image = self.max_pool(image)
            image = self.conv2(image)
            image = self.max_pool(image)
            image = self.conv3(image)
            image = self.max_pool(image)
            return image.reshape((image.shape[0],-1))
        
        x = torch.cat((convolve(image1), convolve(image2)), axis=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        prediction = self.sigmoid(x)
        return prediction

class UWBCategoricalPredictor(torch.nn.Module):
    def __init__(self, input_dim, dims_per_layer, output_dim):
        super().__init__()
        input_dims = [input_dim] + dims_per_layer
        output_dims = dims_per_layer + [output_dim]
        self.linear_layers = torch.nn.ModuleList(torch.nn.Linear(input_dim, output_dim) for input_dim, output_dim in zip(input_dims, output_dims))
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.linear_layers[0](x)

        for linear_layer in self.linear_layers[1:]:
            x = self.relu(x)
            x = linear_layer(x)

        prediction = self.sigmoid(x)
        return prediction