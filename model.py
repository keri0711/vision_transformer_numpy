import numpy as np
from linear_layer import LinearLayer
from position_encoding import position_encoding
from attention import Attention
from activation_function import ActivationFunction
from relu import relu, relu_derivative


class VisionTransformerModel:
    '''
    Vision transformer model.
    '''

    def __init__(self, img_size: tuple[int, int], patch_size: tuple[int, int], pos_emb_size: int, num_classes: int):
        '''
        :param img_size: input image size
        :param patch_size: patch size
        :param pos_emb_size: position embedding vector size
        :param num_classes: number of classes
        '''

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, 'Image size not divisible by patch size!'

        # params
        self.img_size = img_size
        self.patch_size = patch_size
        self.pos_emb_size = pos_emb_size
        self.num_classes = num_classes

        patch_size_sum = patch_size[0] * patch_size[1]
        embedded_patches_size = patch_size_sum + pos_emb_size

        # layers
        self.linear_projection_layer = LinearLayer(input_size=patch_size_sum,
                                                   output_size=patch_size_sum)
        self.attention_layer = Attention(input_size=embedded_patches_size)
        self.mlp_linear_layer = LinearLayer(input_size=embedded_patches_size,
                                            output_size=embedded_patches_size)
        self.relu_layer = ActivationFunction(relu, relu_derivative)
        self.mlp_head = LinearLayer(input_size=embedded_patches_size, output_size=num_classes)

        # extra learnable class embedding
        self.extra_embedding = np.random.rand(1, self.patch_size[0] * self.patch_size[1])

    def forward(self, img: np.array) -> np.array:
        '''
        Forward pass through the model.

        :param img: input image
        :return: class prediction
        '''

        patches = [img[x:x + self.patch_size[0], y:y + self.patch_size[1]] for x in range(0, img.shape[0], self.patch_size[0]) for y in range(0, img.shape[1], self.patch_size[1])]
        patches = np.array(patches)

        # save num patches
        self.num_patches = patches.shape[0]

        # flatten patches
        flattened_patches = np.reshape(patches, (self.num_patches, self.patch_size[0] * self.patch_size[1]))

        # linear projection of flattened patches
        linear_projection = self.linear_projection_layer.forward(flattened_patches)

        # add extra learnable embedding
        linear_projection = np.insert(linear_projection, 0, self.extra_embedding, axis=0)

        # position embedding
        position_embedding_matrix = position_encoding(num_vectors=linear_projection.shape[0],
                                                      vector_size=self.pos_emb_size)

        # patch + position embedding
        embedded_patches = []

        for position, patch in zip(position_embedding_matrix, linear_projection):
            embedded_patches.append(np.concatenate([position, patch]))

        embedded_patches = np.array(embedded_patches)

        # self attention
        attention = self.attention_layer.forward(input=embedded_patches)

        # MLP
        mlp_result = self.mlp_linear_layer.forward(embedded_patches)
        mlp_result = self.relu_layer.forward(mlp_result)

        # transformer encoder result
        encoder_result = attention + mlp_result

        # use extra learnable class embedding for classification
        y_pred = self.mlp_head.forward(np.array([encoder_result[0]]))

        return y_pred

    def backward(self, delta_e_y: np.array, learning_rate: float):
        '''
        Backward pass through the model.

        :param delta_e_y: derivative of the loss function with respect to the outputs
        :param learning_rate: learning rate used to update parameters
        :return: None
        '''

        # mlp head backward
        delta_e_mlp_head = self.mlp_head.backward(delta_e_y=delta_e_y, learning_rate=learning_rate)

        # add zeros because mlp head uses only extra learnable class embedding
        zeros = np.zeros((self.num_patches, delta_e_mlp_head.shape[1]))
        delta_e_mlp_head = np.insert(zeros, 0, delta_e_mlp_head, axis=0)

        # mlp transformer encoder backward
        delta_e_mlp = self.mlp_linear_layer.backward(delta_e_y=delta_e_mlp_head, learning_rate=learning_rate)

        # attention backward
        delta_e_attention = self.attention_layer.backward(delta_e_at=delta_e_mlp, learning_rate=learning_rate)

        # extra learnable class embedding backward
        delta_e_extra = np.array([delta_e_attention[0][self.pos_emb_size:]])
        self.extra_embedding = self.extra_embedding - learning_rate * delta_e_extra

        # linear projection backward
        _ = self.linear_projection_layer.backward(delta_e_y=np.array(delta_e_attention[1:, self.pos_emb_size:]),
                                                  learning_rate=learning_rate)


