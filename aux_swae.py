# Functions required in the notebook ICML_notebook.ipynb

# Libraries to import

import jax.numpy as jnp
import jax
from jax import random
from accountants.main import create_accountant

# GENERATE UNIFORM DATA ON THE SPHERE (INDEPENDENT UNIFORM RADIOUS AND UNIFORM DIRECTION) -----------------------------------------

def sample_unit_ball(key, num_samples, dim):
    key_dir, key_r = random.split(key)
    
    # Step 1: Sample from standard normal
    x = random.normal(key_dir, shape=(num_samples, dim))
    
    # Step 2: Normalize to unit sphere
    x = x / jnp.linalg.norm(x, axis=1, keepdims=True)
    
    # Step 3: Sample radius with proper scaling
    r = random.uniform(key_r, shape=(num_samples,)) ** (1.0 / dim)
    
    # Step 4: Multiply direction by radius
    return x * r[:, None]


# GET THE FIRST INDEXES IN THE DATASET ASSOCIATED WITH AN ELEMENT OF EACH CLASS
def get_first_indexes(label, num_clases):
    def get_first(c):
        return jnp.where(label == c, size=1, fill_value=-1)[0].item() 
    return jnp.array([get_first(c) for c in range(num_clases)])


# COUNT THE NUMBER OF PARAMETERS OF A JAX NEURAL NETWORK 

def count_params_by_layer(params):
    total_params = 0
    for param_name, param_value in params.items():
        param_count = param_value.size  # Number of elements in the parameter tensor
        total_params += param_count
        print(f"{param_name},\t  {param_count}")
    
    print(f"\nTotal \t \t {total_params}")


# COMPUTE THE MATRIX R_ij IN A SPARSE FORM: IT RETURNS THE INDEXES WITH R_ij>0 AND THE VALUE ON THOSE INDEXES -----------------
def R_and_indexes(n0, n1):

    if n0==n1:
        R_vector =  jnp.full((n0,), 1.0 / n0)
        aux = jnp.arange(0, n0).reshape(n0, 1)
        indexes = jnp.tile(aux, (1, 2))

    else:
        # Compute the matrix R (in a vectorized form, to save memory)
        indexes = jnp.zeros((2 * max(n0, n1), 2), dtype=jnp.int32)  # indexes (i, j) with R(i, j) > 0
        R_vector = jnp.zeros(2 * max(n0, n1), dtype=jnp.float32)  # values R(i, j) for (i, j) in indexes
    
        j = 0
        index = 0
        for i in range(n0):
            while (j / n1) < ((i + 1) / n0):
                indexes = indexes.at[index].set(jnp.array([i, j]))
                R_vector = R_vector.at[index].set(
                    min((i + 1) / n0, (j + 1) / n1) - max(i / n0, j / n1)
                )
                index += 1
                j += 1
            j -= 1
    
        indexes = indexes[:index]
        R_vector = R_vector[:index]
        
    return R_vector, indexes


# VALUE AND GRADIENT OF THE ONE-DIMENSIONAL WASSERSTEIN DISTANCE ---------------------------------------------------------------


def wasserstein_1d(U, V, R_vector, indexes, return_gradient=True):

    if U.ndim == 1:  # In this case, we are computing the wasserstein distance between two 1d vectros   
        n0 = U.shape[0]
        n1 = V.shape[0]
    
        # Sort X and Y along the projection dimension
        U_sort = jnp.sort(U, axis=0)
        V_sort = jnp.sort(V, axis=0)
        U_rank = jnp.argsort(U, axis=0)
        V_rank = jnp.argsort(V, axis=0)
        
        # Reconstruct indices for gradients
        reconstruct_U = jnp.argsort(U_rank, axis=0)
        reconstruct_V = jnp.argsort(V_rank, axis=0)
    
    
        # Expand sorted values based on indexes
        U_sort_expanded = U_sort[indexes[:, 0]]  # Shape: (num_pairs, num_projections)
        V_sort_expanded = V_sort[indexes[:, 1]]  # Shape: (num_pairs, num_projections)
    
        # Compute differences and squared differences
        differences = U_sort_expanded - V_sort_expanded  # Shape: (num_pairs, num_projections)
        squared_differences = differences**2  # Shape: (num_pairs, num_projections)
    
        # Compute Wasserstein distances
        wass_dist = jnp.dot(squared_differences.T, R_vector)  # Shape: (num_projections,)

        if return_gradient == False:
            return wass_dist, None, None 
            
        else: 
    
            # Compute the gradient w.r.t. U
            weighted_differences = 2 * differences * R_vector  # Shape: (num_pairs, num_projections)
            wass_grad_sort_U = jnp.zeros(n0, dtype=differences.dtype)
            wass_grad_sort_U = wass_grad_sort_U.at[indexes[:, 0]].add(weighted_differences)
        
            # Compute the gradient w.r.t. V
            wass_grad_sort_V = jnp.zeros(n1, dtype=differences.dtype)
            wass_grad_sort_V = wass_grad_sort_V.at[indexes[:, 1]].add(-weighted_differences)
        
            # Reorder the gradient to obtain the gradient w.r.t. the original variables
            wass_grad_U = wass_grad_sort_U[reconstruct_U]
            wass_grad_V = wass_grad_sort_V[reconstruct_V]
        
            return wass_dist, wass_grad_U, wass_grad_V

    else:
        n0, num_projections = U.shape
        n1 = V.shape[0]
    
        # Sort X and Y along the projection dimension
        U_sort = jnp.sort(U, axis=0)
        V_sort = jnp.sort(V, axis=0)
        U_rank = jnp.argsort(U, axis=0)
        V_rank = jnp.argsort(V, axis=0)
        
        # Reconstruct indices for gradients
        reconstruct_U = jnp.argsort(U_rank, axis=0)
        reconstruct_V = jnp.argsort(V_rank, axis=0)
    
    
        # Expand sorted values based on indexes
        U_sort_expanded = U_sort[indexes[:, 0], :]  # Shape: (num_pairs, num_projections)
        V_sort_expanded = V_sort[indexes[:, 1], :]  # Shape: (num_pairs, num_projections)
    
        # Compute differences and squared differences
        differences = U_sort_expanded - V_sort_expanded  # Shape: (num_pairs, num_projections)
        squared_differences = differences**2  # Shape: (num_pairs, num_projections)
    
        # Compute Wasserstein distances
        wass_dist = jnp.dot(squared_differences.T, R_vector)  # Shape: (num_projections,)
    
        if return_gradient==False: 
            return wass_dist, None, None 
        else:
            # Compute the gradient w.r.t. U
            weighted_differences = 2 * differences * R_vector[:, None]  # Shape: (num_pairs, num_projections)
            wass_grad_sort_U = jnp.zeros((n0, num_projections), dtype=differences.dtype)
            wass_grad_sort_U = wass_grad_sort_U.at[indexes[:, 0],].add(weighted_differences)
        
            # Compute the gradient w.r.t. V
            wass_grad_sort_V = jnp.zeros((n1, num_projections), dtype=differences.dtype)
            wass_grad_sort_V = wass_grad_sort_V.at[indexes[:, 1],].add(-weighted_differences)
        
            # Reorder the gradient to obtain the gradient w.r.t. the original variables
            wass_grad_U = wass_grad_sort_U[reconstruct_U, jnp.arange(num_projections)]
            wass_grad_V = wass_grad_sort_V[reconstruct_V, jnp.arange(num_projections)]
        
            return wass_dist, wass_grad_U, wass_grad_V



# VALUE AND GRADIENT OF THE SLICED WASSERSTEIN DISTANCE -----------------------------------------------------------------------

def sliced_wasserstein(U, V, R_vector, indexes, random_directions, return_gradient = True):
    """
    Computes the Sliced Wasserstein Distance in batch for multiple random projections.
    """
    n0, dim = U.shape
    num_projections = random_directions.shape[0]

    # Project U and V onto all random directions simultaneously
    U_proj = jnp.dot(U, random_directions.T)
    V_proj = jnp.dot(V, random_directions.T)

    # Compute the Wasserstein distance and gradients in batch for all projections
    wass_dists, wass_grads_U, wass_grads_V = wasserstein_1d(U_proj, V_proj, R_vector, indexes, return_gradient)

    # Average sliced Wasserstein distance
    swd_dist = jnp.mean(wass_dists)

    if return_gradient == False: 
        return swd_dist, None, None
    else: 
        # Compute the average gradient over all projections
        swd_grad_U = jnp.dot(wass_grads_U, random_directions) / num_projections
        swd_grad_V = jnp.dot(wass_grads_V, random_directions) / num_projections
    
        return swd_dist, swd_grad_U, swd_grad_V


# BINARY CROSS ENTROPY LOSS -------------------------------------------------------------------------------------------------

def bce_loss(params, model, x):
    x_reconstructed = model.forward(x, params)
    x_reconstructed = jnp.clip(x_reconstructed, 1e-6, 1 - 1e-6)
    loss = -jnp.mean(x * jnp.log(x_reconstructed) + (1 - x) * jnp.log(1 - x_reconstructed))
    return loss

# LIMITS TO DIVIDE A BATCH IN MINIBATCHES, TO ALLEVIATE COMPUTATIONS ---------------------------------------------------------

def minibatch_limits(batch_size, minibatch):
    return (jnp.arange((batch_size-1)//minibatch)+1)*minibatch


# FIND NOISE REQUIRED TO ACHIEVE A GIVEN (EPSILON, DELTA) PRIVACY BUDGET ------------------------------------------------------
def find_noise_for_epsilon(num_iterations, delta, sampling_rate, target_epsilon, tolerance=1e-4):
    """
    Calculate the noise multiplier needed to achieve (epsilon, delta)-DP.

    Args:
        num_iterations (int): Number of iterations.
        delta (float): Desired delta value for DP.
        sampling_rate (float): Sampling rate of the data.
        target_epsilon (float): Target epsilon value for DP.
        tolerance (float): Tolerance for the noise multiplier search.

    Returns:
        float: The noise multiplier that achieves (epsilon, delta)-DP.
    """
    # Initialize the accountant
    accountant = create_accountant("gdp")

    # Binary search for the noise multiplier
    lower_bound = 0.001  # Start with a small noise multiplier
    upper_bound = 100.0  # Arbitrary upper limit for noise multiplier
    best_noise = None

    while upper_bound - lower_bound > tolerance:
        current_noise = (lower_bound + upper_bound) / 2

        # Reset the accountant for each test
        accountant = create_accountant("gdp")

        # Simulate the accountant steps with the current noise multiplier
        for _ in range(num_iterations):
            accountant.step(noise_multiplier=current_noise, sample_rate=sampling_rate)

        # Calculate the resulting epsilon
        epsilon = accountant.get_epsilon(delta=delta, poisson=False)

        # Adjust bounds based on comparison with target_epsilon
        if epsilon > target_epsilon:
            lower_bound = current_noise  # Increase noise
        else:
            upper_bound = current_noise  # Decrease noise
            best_noise = current_noise
    return best_noise

# XAVIER UNIFORM INITIALIZATION OF THE PARAMETERS ---------------------------------------------------------------------------

def xavier_uniform(key, shape, gain=1.0):
    fan_in, fan_out = shape[0], shape[1]
    limit = gain * jnp.sqrt(6 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)


# CLIP THE ABSOLUTE VALUE OF THE ENTRIES OF A VECTOR, OR THE NORM OF THE ROWS OF A MATRIX ----------------------------------
def clip_row_norms(matrix_or_vector, max_norm):
    """
    Clips the norms of rows of a matrix or a vector to a maximum value.

    Args:
        matrix_or_vector (jnp.ndarray): The input matrix or vector.
        max_norm (float): The maximum norm for the rows or the vector.

    Returns:
        jnp.ndarray: The matrix or vector with norms clipped.
    """
    # Check if the input is a matrix or a vector
    if matrix_or_vector.ndim == 1:
        return jnp.clip(matrix_or_vector, -max_norm, max_norm)
    else:
        # Handle the case for a matrix
        matrix = matrix_or_vector
        
        # Compute the norms of each row
        row_norms = jnp.linalg.norm(matrix, axis=1, keepdims=True)

        # Compute the clipping factors
        clipping_factors = jnp.minimum(1.0, max_norm / (row_norms + 1e-8))

        # Scale the rows of the matrix
        clipped_matrix = matrix * clipping_factors

        return clipped_matrix


####################################################  FUNCTIONS TO COMPUTE GRADIENTS WITH THE OUTPUTS OF NEURAL NETWORKS ########################################################

# KEY COMMENTS 

#  -  For simplicity in the inner clipping process, both parameters and output of Neural networks are flattened
#  -  Within each function, we need to compute the jacobian matrix of the last layer (or intermediate layer in autoencoder) of the NN. This is very high dimensional, size  n x nº params x output dimension
#  -  Despite of the fact that we are appling DPSGD, we parallelize the computation of this jacobian even within each batch, dividing each batch in k minibatches (10 by default)
#  -  This minibatch strategy forces to select batch sizes with suitable number of samples, such that can be exactly divided in smaller minibatches. 
#  -  Aggregation is then performed either parallelizing with jax.vmap, which is faster but requires a large amount of memory, or without parallelizing, with another loop, which is much slower but avoids running out of memory. 

#################################################################################################################################################################################


# RECONSTRUCTION GRADIENT UPDATES FOR THE MODEL ---------------------------------------------------------------------------------------

def gradient_reconstruction_loss(flat_params, unflatten_params, X_batch, model, C, minibatch_number = 10, parallel = False):

    # Loss function: Binary Cross-Entropy
    def individual_mse_loss_flat(flat_params, unflatten_params, model, X):
        X_pred = model.forward(X, unflatten_params(flat_params))
        return jnp.sum((X_pred - X)**2, axis=tuple(range(1, X.ndim)))
        #return jnp.sum( (X_pred - X)**2 , axis=1)
        #return jnp.sqrt(jnp.sum( (Y_pred - Y)**2 , axis=1))
    
    def individual_bce_loss_flat(flat_params, unflatten_params, model, X):
        X_pred = model.forward(X, unflatten_params(flat_params))
        # Asegúrate de que X_pred esté en el rango (0,1), normalmente con sigmoid ya está
        eps = 1e-6  # para evitar log(0)
        X_pred = jnp.clip(X_pred, eps, 1 - eps)
        return jnp.mean(-X * jnp.log(X_pred) - (1 - X) * jnp.log(1 - X_pred), axis=tuple(range(1, X.ndim)))

                
    jacobian_loss= jax.jacobian(individual_bce_loss_flat, argnums=0)
      
    def compute_gradient_reconstruction_minibatch(flat_params, unflatten_params, X_minibatch, C):
        # Calcula el gradiente para el minibatch
        gradient_loss_flat_params = jacobian_loss(flat_params, unflatten_params, model, X_minibatch)
        gradient_loss_flat_params_clipped = clip_row_norms(gradient_loss_flat_params, C)
        gradient = gradient_loss_flat_params_clipped.mean(axis=0)
        return gradient

    vmap_gradients_loss = jax.vmap(compute_gradient_reconstruction_minibatch, in_axes=(None, None, 0, None))

    batch_size = X_batch.shape[0]
    minibatch = batch_size // minibatch_number
    minibatch_limit = minibatch_limits(batch_size, minibatch)
    
    X_minibatch_array = jnp.array(jnp.split(X_batch, minibatch_limit, axis=0))
    
    if parallel == False:
        gradient_sum = 0
        for i in range(len(X_minibatch_array)):
            X_minibatch = X_minibatch_array[i]
            grad = compute_gradient_reconstruction_minibatch(flat_params, unflatten_params, X_minibatch, C)
            gradient_sum += grad
        gradient_flat_params = gradient_sum / len(X_minibatch_array)
    else:
        gradient_flat_params = vmap_gradients_loss(flat_params, unflatten_params, X_minibatch_array, C).mean(axis=0)     
   
    return  gradient_flat_params




## SLICED WASSERSTEIN GRADIENT UPDATES FOR THE MODEL  ----------------------------------------------------------------------------------------------------------

def gradient_sliced_flow(flat_params, unflatten_params, X_batch, Z_batch, model, R_vector_batch, indexes_batch, M, L, random_directions=1, minibatch_number = 10, parallel=False):
    # Function to compute the jacobian of the last NN layer w.r.t. the flatten parameters
    def last_layer(flat_params, unflatten_params, X_batch):
        U = model.penalized_layer(X_batch, unflatten_params(flat_params))
        return jnp.ravel(U)
    grad_last_layer = jax.jacobian(last_layer, argnums=0)
    
    
    def compute_gradient_sliced_minibatch(flat_params, unflatten_params, X_minibatch, swd_grad_flat_minibatch,L):
        # Calcula el gradiente para el minibatch
        gradient_last_layer_flat_params = grad_last_layer(flat_params, unflatten_params, X_minibatch)
        gradient_last_layer_flat_params_clipped = clip_row_norms(gradient_last_layer_flat_params, L)
        gradient = jnp.dot(swd_grad_flat_minibatch, gradient_last_layer_flat_params)
        return gradient

    # Apply vmap over the minibatches
    vmap_gradients = jax.vmap(compute_gradient_sliced_minibatch, in_axes=(None, None, 0, 0, None))

    batch_size_X = X_batch.shape[0]
    minibatch_X = batch_size_X//minibatch_number
    minibatch_limits_X = minibatch_limits(batch_size_X, minibatch_X)

    # Forward pass
    U_batch = model.penalized_layer(X_batch, unflatten_params(flat_params))
    V_batch = Z_batch

    # Forward pass - clipped values 
    U_batch_clipped = clip_row_norms(U_batch, M)
    V_batch_clipped = clip_row_norms(V_batch, M)

    # Compute Sliced Wasserstein distance and gradient
    if U_batch.ndim ==1: 
        d = 1
        swd_dist, swd_grad_U, swd_grad_V =  wasserstein_1d(U_batch_clipped, V_batch_clipped, R_vector_batch, indexes_batch)
    else: 
        d = U_batch.shape[1]
        swd_dist, swd_grad_U, swd_grad_V = sliced_wasserstein(U_batch_clipped, V_batch_clipped, R_vector_batch, 
                                                                  indexes_batch, random_directions)
        
    swd_grad_flat_U = jnp.ravel(swd_grad_U)

    swd_grad_flat_U_minibatch_array = jnp.array(jnp.split(swd_grad_flat_U, d*minibatch_limits_X))
    X_minibatch_array = jnp.array(jnp.split(X_batch, minibatch_limits_X, axis=0))

    
    if parallel == False:
        gradient_sum = 0
        for i in range(len(X_minibatch_array)):
            X_minibatch = X_minibatch_array[i]
            swd_grad_flat_minibatch = swd_grad_flat_U_minibatch_array[i]
            grad = compute_gradient_sliced_minibatch(flat_params, unflatten_params, X_minibatch, swd_grad_flat_minibatch,L)
            gradient_sum += grad
        gradient_flat_params = gradient_sum
    else:
        gradient_flat_params = vmap_gradients(flat_params, unflatten_params, X_minibatch_array, swd_grad_flat_U_minibatch_array,L).sum(axis=0)
     
    return gradient_flat_params



