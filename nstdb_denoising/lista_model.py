"""
CNN-LISTA Deep Unrolled Model for ECG Patch Denoising.

Replaces classical OMP sparse coding with a Learned Iterative
Shrinkage-Thresholding Algorithm (LISTA) implemented as a
TensorFlow/Keras Model subclass.

Architecture:
  Encoder   : Conv1D  (patch → sparse domain, trainable)
  Update    : Dense   (residual refinement, trainable)
  Decoder   : Dense   (sparse code → patch, FROZEN sklearn dict)
  Threshold : Per-iteration learnable soft thresholding

Forward pass (3 unrolled iterations):
  α₀ = S_λ₀(W_e · y)
  αₜ = S_λₜ(αₜ₋₁ + W_g · W_e(y − D·αₜ₋₁))   for t = 1..T-1
  output = D · α_final
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy import signal as sp_signal
import joblib

class SoftThreshold(layers.Layer):
    """Learnable soft-thresholding — the sparsity-forcing nonlinearity."""

    def __init__(self, initial_lambda=0.1, **kwargs):
        super().__init__(**kwargs)
        self.initial_lambda = initial_lambda

    def build(self, input_shape):
        self.threshold = self.add_weight(
            name="lambda",
            shape=(1,),
            initializer=tf.constant_initializer(self.initial_lambda),
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        return tf.sign(x) * tf.maximum(tf.abs(x) - self.threshold, 0.0)

    def get_config(self):
        config = super().get_config()
        config['initial_lambda'] = self.initial_lambda
        return config

class LISTADenoisingModel(Model):
    """
    Deep Unrolled CNN-LISTA for supervised ECG patch denoising.

    Maps noisy zero-mean patches (N, 64, 1) → clean patches (N, 64).
    Decoder is frozen to the sklearn dictionary; encoder, update
    layer, and per-iteration thresholds are trainable.
    """

    def __init__(self, input_dim=64, num_atoms=64, iterations=3,
                 dictionary_weights=None, sparsity_penalty=0.001,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.num_atoms = num_atoms
        self.iterations = iterations
        self.sparsity_penalty = sparsity_penalty

        self.encoder = layers.Conv1D(
            filters=num_atoms, kernel_size=input_dim,
            padding='valid', use_bias=False, name='encoder_conv',
        )

        self.update_layer = layers.Dense(
            num_atoms, use_bias=False, name='update_dense',
        )

        self.thresholds = [
            SoftThreshold(name=f'soft_threshold_{i}')
            for i in range(iterations)
        ]

        self.decoder = layers.Dense(
            input_dim, use_bias=False, name='decoder_dense',
        )

        if dictionary_weights is not None:
            self.decoder.build(input_shape=(None, num_atoms))
            self.decoder.set_weights([dictionary_weights.T])
            self.decoder.trainable = True

    def call(self, y, training=False):
        """
        3-iteration unrolled ISTA forward pass.

        Parameters
        ----------
        y : tf.Tensor, shape (batch, input_dim, 1)
            Conv1D-formatted noisy patches.

        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            (clean_reconstruction (batch, input_dim),
             alpha_sparse_codes   (batch, num_atoms))
        """
        z = self.encoder(y)              
        z = tf.squeeze(z, axis=1)        
        alpha = self.thresholds[0](z)

        for i in range(1, self.iterations):
            recon = self.decoder(alpha)                            
            error = tf.squeeze(y, axis=-1) - recon                
            enc_err = self.encoder(tf.expand_dims(error, -1))     
            update = self.update_layer(enc_err[:, 0, :])         
            alpha = self.thresholds[i](alpha + update)

        clean_reconstruction = self.decoder(alpha)
        return clean_reconstruction, alpha

    def train_step(self, data):
        """Custom training step: MSE reconstruction + L1 sparsity + grad clipping."""
        x, y_true = data

        with tf.GradientTape() as tape:
            recon, alpha = self(x, training=True)
            mse_loss = tf.reduce_mean(tf.square(y_true - recon))
            l1_loss = self.sparsity_penalty * tf.reduce_mean(tf.abs(alpha))
            total_loss = mse_loss + l1_loss

        grads = tape.gradient(total_loss, self.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return {'loss': total_loss, 'mse': mse_loss, 'l1_sparsity': l1_loss}

    def test_step(self, data):
        """Custom validation step."""
        x, y_true = data
        recon, alpha = self(x, training=False)
        mse_loss = tf.reduce_mean(tf.square(y_true - recon))
        l1_loss = self.sparsity_penalty * tf.reduce_mean(tf.abs(alpha))
        total_loss = mse_loss + l1_loss
        return {'loss': total_loss, 'mse': mse_loss, 'l1_sparsity': l1_loss}

def learn_initial_dictionary(clean_patches, n_components=64, max_iter=50):
    """
    Train sklearn MiniBatchDictionaryLearning to initialize LISTA decoder.

    Parameters
    ----------
    clean_patches : np.ndarray, shape (N, 64)
        Zero-mean clean patches.
    n_components : int
        Number of dictionary atoms.
    max_iter : int
        Training iterations.

    Returns
    -------
    np.ndarray, shape (n_components, 64)
        Dictionary matrix (rows = atoms).
    """
    from sklearn.decomposition import MiniBatchDictionaryLearning

    print(f'  Learning sklearn dictionary: {n_components} atoms, '
          f'{max_iter} iters...')
    dl = MiniBatchDictionaryLearning(
        n_components=n_components,
        transform_algorithm='omp',
        transform_n_nonzero_coefs=3,
        batch_size=256,
        max_iter=max_iter,
        n_jobs=-1,
        random_state=42,
        verbose=0,
    )
    dl.fit(clean_patches)
    dictionary = dl.components_
    norms = np.linalg.norm(dictionary, axis=1)
    print(f'  Dictionary shape: {dictionary.shape}')
    print(f'  Atom norms: [{norms.min():.3f}, {norms.max():.3f}]')
    return dictionary


def prepare_patches_for_lista(noisy_patches_zm, clean_patches_zm=None):
    """
    Reshape (N, 64) patches to Conv1D format (N, 64, 1) float32.

    Parameters
    ----------
    noisy_patches_zm : np.ndarray, shape (N, 64)
    clean_patches_zm : np.ndarray or None, shape (N, 64)

    Returns
    -------
    tuple
        X: (N, 64, 1) float32, Y: (N, 64) float32 or None
    """
    X = noisy_patches_zm.astype(np.float32)[..., np.newaxis]
    Y = (clean_patches_zm.astype(np.float32)
         if clean_patches_zm is not None else None)
    return X, Y


def build_lista_model(dictionary_weights, input_dim=64, num_atoms=64,
                      iterations=3, sparsity_penalty=0.001, lr=1e-3,
                      total_steps=None):
    """
    Build, initialize, compile, and return a ready-to-train LISTA model.

    Parameters
    ----------
    dictionary_weights : np.ndarray, shape (num_atoms, input_dim)
        sklearn dictionary to freeze into the decoder.
    input_dim : int
        Patch length (64).
    num_atoms : int
        Number of atoms / sparse code dimension.
    iterations : int
        Unrolled ISTA iterations.
    sparsity_penalty : float
        L1 weight on alpha.
    lr : float
        Initial Adam learning rate.
    total_steps : int or None
        Total training steps for cosine decay. If None, uses flat LR.

    Returns
    -------
    LISTADenoisingModel
        Compiled model ready for model.fit().
    """
    model = LISTADenoisingModel(
        input_dim=input_dim,
        num_atoms=num_atoms,
        iterations=iterations,
        dictionary_weights=dictionary_weights,
        sparsity_penalty=sparsity_penalty,
    )

    dummy = tf.zeros((1, input_dim, 1), dtype=tf.float32)
    _ = model(dummy)

    if total_steps is not None and total_steps > 0:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=total_steps,
            alpha=lr / 20,  
        )
        print(f'  LR schedule: CosineDecay {lr} → {lr/20:.1e} '
              f'over {total_steps:,} steps')
    else:
        lr_schedule = lr

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    )
    model.summary()

    return model

def denoise_signal_lista(noisy_hp_signal, lista_model, window_size=64,
                         savgol_window=11, savgol_polyorder=3,
                         batch_size=512, verbose=False):
    """
    Denoise an HP-filtered signal using the LISTA model.

    Drop-in replacement for denoise_signal() — same zero-means
    reconstruction strategy (patch means discarded to prevent
    EM artifact re-injection).

    Parameters
    ----------
    noisy_hp_signal : np.ndarray
        HP-filtered noisy ECG signal.
    lista_model : LISTADenoisingModel
        Trained LISTA model.
    window_size : int
        Patch length (must match model's input_dim).
    savgol_window : int
        Savitzky-Golay post-processing window.
    savgol_polyorder : int
        Savitzky-Golay polynomial order.
    batch_size : int
        Prediction batch size.
    verbose : bool
        Print patch count.

    Returns
    -------
    np.ndarray
        Denoised signal.
    """
    from .dictionary import extract_dense_patches

    n = len(noisy_hp_signal)
    patches_zm, _, n_patches = extract_dense_patches(
        noisy_hp_signal, window_size,
    )

    if verbose:
        print(f'    LISTA patches: {n_patches:,} (stride=1)')

    X = patches_zm.astype(np.float32)[..., np.newaxis]

    outputs = lista_model.predict(X, batch_size=batch_size, verbose=0)
    recon_patches = (outputs[0] if isinstance(outputs, (list, tuple))
                     else outputs)

    rec = np.zeros(n)
    cnt = np.zeros(n)
    for p in range(n_patches):
        end = min(p + window_size, n)
        actual = end - p
        rec[p:end] += recon_patches[p, :actual]
        cnt[p:end] += 1

    cnt[cnt == 0] = 1
    rec /= cnt

    rec = sp_signal.savgol_filter(
        rec, window_length=savgol_window, polyorder=savgol_polyorder,
    )
    return rec

def save_lista_model(model, dictionary, config, model_dir):
    """
    Save LISTA model weights + sklearn dictionary + config.

    Creates:
      model_dir/lista_weights.weights.h5   — trained TF weights
      model_dir/lista_config.pkl           — dictionary + metadata
    """
    os.makedirs(model_dir, exist_ok=True)

    weights_path = os.path.join(model_dir, 'lista_weights.weights.h5')
    model.save_weights(weights_path)

    config_path = os.path.join(model_dir, 'lista_config.pkl')
    joblib.dump({
        'dictionary':      dictionary,
        'input_dim':       model.input_dim,
        'num_atoms':       model.num_atoms,
        'iterations':      model.iterations,
        'sparsity_penalty': model.sparsity_penalty,
        'window_size':     config.get('window_size', 64),
        'hp_cutoff':       config.get('hp_cutoff', 0.67),
        'hp_order':        config.get('hp_order', 4),
        'savgol_window':   config.get('savgol_window', 11),
        'savgol_polyorder': config.get('savgol_polyorder', 3),
        'fs':              config.get('fs', 360),
        'config':          config,
    }, config_path, compress=3)

    print(f'  LISTA weights saved: {weights_path}')
    print(f'  LISTA config saved : {config_path}')

def load_lista_model(model_dir):
    """
    Load a pre-trained LISTA model from disk.

    Parameters
    ----------
    model_dir : str
        Directory containing lista_weights.weights.h5 and lista_config.pkl.

    Returns
    -------
    tuple[LISTADenoisingModel, dict]
        (model, artifact_dict)
    """
    config_path = os.path.join(model_dir, 'lista_config.pkl')
    artifact = joblib.load(config_path)

    print(f'  Loading LISTA model from: {model_dir}')

    model = LISTADenoisingModel(
        input_dim=artifact['input_dim'],
        num_atoms=artifact['num_atoms'],
        iterations=artifact['iterations'],
        dictionary_weights=artifact['dictionary'],
        sparsity_penalty=artifact['sparsity_penalty'],
    )

    dummy = tf.zeros((1, artifact['input_dim'], 1), dtype=tf.float32)
    _ = model(dummy)

    weights_path = os.path.join(model_dir, 'lista_weights.weights.h5')
    model.load_weights(weights_path)

    print(f'  Architecture: {artifact["input_dim"]}→{artifact["num_atoms"]} '
          f'atoms, {artifact["iterations"]} iterations')

    return model, artifact