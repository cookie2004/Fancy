import retro
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

import numpy as np
import cv2

import matplotlib.pylab as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from IPython.display import clear_output

def layers_output(model, X):
    """
    Captures and returns the outputs of each layer in a Keras model given an input.

    Args:
        model (keras.Model): A Keras model.
        X (numpy.ndarray or tf.Tensor): The input to the model.

    Returns:
        dict: A dictionary where keys are layer names and values are lists containing:
            - The layer's class name (as a string).
            - The output of the layer (as a NumPy array or TensorFlow Tensor).

    Raises:
        ValueError: If the input is not a NumPy array or TensorFlow Tensor.
        TypeError: If the model is not a Keras model.
    """
    if len(X.shape) == 3:
        X = np.expand_dims(X, axis=0)
    if not isinstance(model, keras.Model):
        raise TypeError("model must be a keras.Model instance")

    if not isinstance(X, (np.ndarray, tf.Tensor)):
        raise ValueError("X must be a NumPy array or TensorFlow Tensor")
        
    output = {model.layers[0].name:[model.layers[0].__class__.__name__, np.squeeze(X)]}
    for i in model.layers[1:]:
        X = i(X)
        output[i.name] = [i.__class__.__name__, np.squeeze(X)]
    return output

def epsilon_greedy(actionprobs, epsilon_greedy):
    #Choose epsilon greedy action.
    if np.random.random() < epsilon:
        a = np.random.choice(np.arange(len(actionprobs)))
    else: 
        a = tf.argmax(a_probs).numpy()
            
def collect_episode(env, model, preprocessore):
    #Create video object. We will write the frames to this object.
    #out = cv2.VideoWriter(output_name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60/frameskips, (1800,1440))
    #You have to reset the environment to the start of the level.
    states = []
    rewards = []
    
    s = env.reset()
    
    #Prime the frame stack.
    s, episode_return, done, info = preprocessor(action_space[0], env, frameskips)
    episode_return = 0
    return_list = [0]
    done = False
    lives = int(info['lives'])
    while not(done):
        states.append([s,raw_obs])
        #Choose epsilon greedy action.
        if np.random.random() < epsilon:
            a = np.random.choice(np.arange(len(action_space)))
        else: 
            a_probs = model(np.expand_dims(s,0), training=False)
            a = tf.argmax(a_probs[0]).numpy()
    
        #Collect information on next state.
        s, reward, done, info = preprocessor(action_space[a], env, frameskips)
        raw_obs = env.get_screen()[:, :, [0, 1, 2]]
        rewards.append(reward)
        return_list.append(episode_return)
        
    #Release the video object. 
    #out.release()
    #clear_output()
    print ('Evaluation complete.')
    return states, rewards

def visualize_image(array, ax, title, im_per_row = 16, linear_to_image_rows = 10):
    ax.axis('off')
    ax.set_title(title + str(array.shape), color='white')
    assert len(array.shape) < 4, "Input array dimensions must be < 4."
    if len(array.shape)==1:
        array = np.concatenate((array, np.ones(linear_to_image_rows - len(array)%linear_to_image_rows)))
        array = array.reshape((linear_to_image_rows, int(len(array)/linear_to_image_rows)))
        array = (array/np.max(array)) * 255
        ax.imshow(array, cmap='viridis')
        
    elif len(array.shape)==2:
        ax.imshow(array, cmap='viridis')
    elif len(array.shape)==3 and array.shape[2]==3:
        ax.imshow(array)
    elif len(array.shape)==3 and array.shape[2]>3:
        for image in range(array.shape[2]):
            #ax.imshow(array[:,:,image], origin='upper', extent=[image*array.shape[0],image*array.shape[1]+array.shape[1], 0, array.shape[1]])
            #ax.plot([image*84,image*84],[0,84],color='black')
            ax.imshow(array[:,:,image], origin='upper',extent=[int(image%im_per_row)*array.shape[0],int(image%im_per_row)*array.shape[0]+array.shape[1], int(image/im_per_row)*array.shape[1], int(image/im_per_row)*array.shape[1]+array.shape[1]])
            ax.plot([int(image%im_per_row)*array.shape[0],int(image%im_per_row)*array.shape[1]],[int(image/im_per_row)*array.shape[0],int(image/im_per_row)*array.shape[1]+array.shape[1]],linewidth=0.1,color='black')
    #return ln
#i*array.shape[0],i*array.shape[1]+array.shape[1]

    
def create_layout(model):
    #Get number of hidden layers. 

    num_hidden = len([i for i in model.layers[0:-1]])
    
    fig = plt.figure(figsize=(8, 8), constrained_layout=True)
    fig.patch.set_facecolor('black')
    gs = fig.add_gridspec(np.max((num_hidden, 6)), 7)
    
    raw = fig.add_subplot(gs[:int(np.max((num_hidden, 6))/2),:4])
    layer_figs = [fig.add_subplot(gs[i, 4:]) for i in range(np.max((num_hidden, 3)))]
    model_output = fig.add_subplot(gs[int(np.max((num_hidden, 6))/2):, 0:2])
    reward = fig.add_subplot(gs[int(np.max((num_hidden, 6))/2):, 2:4])
    
    raw.axis('off')
    
    return raw, layer_figs, model_output, reward, fig 

def decision_figure(model_output, ax):
    ax.bar(np.arange(len(model_output)), model_output,edgecolor='white', color='black')
    ax.set_ylabel('Q value', fontsize=10, color='white')
    ax.set_xlabel('Action', fontsize=10, color='white')
    ax.tick_params(axis='both', labelsize=10, color='white', labelcolor='white')
    ax.set_facecolor('black')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    
def reward_plot(reward, ax):
    ax.plot(reward, color='white')
    ax.set_ylabel('Cumulative Reward', fontsize=10, color='white')
    ax.set_xlabel('Step', fontsize=10, color='white')
    ax.tick_params(axis='x', labelsize=10, color='white', labelcolor='white')
    ax.tick_params(axis='y', which='both', labelleft=False)
    ax.set_facecolor('black')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white') 

def create_fancy_video(ep, model, output_name='output', video_dims = (800, 800)):
    out = cv2.VideoWriter(output_name+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (800,800))
    running_reward = []
    for i, state in enumerate(ep[0]):

        running_reward.append(np.sum(ep[1][0:i]))

        layer_outs = layers_output(model, state[0])
        layers = list(layer_outs.keys())

        raw, layer_figs, model_output_fig, reward_fig, fig = create_layout(model)

        #Raw observation.
        raw.imshow(state[1])

        #Plot the hidden layer data.
        for j, layer in enumerate(layer_figs):
            visualize_image(layer_outs[layers[j]][1], layer, list(layer_outs.values())[j][0])

        #Plot final output of model in bar chart.    
        decision_figure(layer_outs[layers[-1]][1], model_output_fig)

        #Plot running reward data.
        reward_plot(running_reward, reward_fig)

        #These lines convert the matplotlib figure into a numpy image array. 
        canvas = FigureCanvas(fig)
        canvas.draw()  
        width, height = fig.get_size_inches() * fig.get_dpi()
        image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)
        out.write(image[:, :, [2, 1, 0]])

    out.release()
    clear_output()