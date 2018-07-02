from mnist import MNIST
import numpy
import time
import NeuralNetwork

#Création du Neural Network
digit_network = NeuralNetwork.NeuralNetwork()




#Ouverture des donnée MNIST
mndata = MNIST("./")

image_training, label_training = mndata.load_training()
image_testing, label_testing = mndata.load_testing()


#Lorsque l'on souhaite train/guess une image, on la convertit d'abord en nombre entre 0 et 1
#Boucle qui prépare et envoi les donnés au training
for i in range(0,3000):
    image_a_transformer = image_training[i]
    image_transformer = [[pixel / 255] for pixel in image_a_transformer]

    label_a_transformer = label_training[i]
    array_label = [0,0,0,0,0,0,0,0,0,0]
    array_label[label_a_transformer] = 1
    array_label = [[element] for element in array_label]
    label_transformer = numpy.asmatrix(array_label)

    digit_network.train(image_transformer,label_transformer)



#Boucle qui prépare l'Envoi du test set
#Display égalment des statistiques
for i in range(0,10):
    image_a_transformer = image_testing[i]
    image_transformer = [[pixel / 255] for pixel in image_a_transformer]

    label = label_testing[i]

    guess = numpy.round(digit_network.guess(image_transformer),2)

    print(mndata.display(image_a_transformer))
    print("Guess:",guess)
    time.sleep(0.5)

