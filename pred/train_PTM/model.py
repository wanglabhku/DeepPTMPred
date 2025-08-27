import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, Dense, Embedding, Dropout, LayerNormalization,
    Conv1D, GlobalAveragePooling1D, Input, Add,
    Concatenate, BatchNormalization, Activation,
    Flatten, RepeatVector, Permute, Multiply, Lambda,
    SpatialDropout1D
)
from config import Config
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow_addons.optimizers import AdamW
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
# ==================== 模型组件 ====================
class PositionEmbedding(Layer):
    def __init__(self, max_len=100, embed_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.embed_dim = embed_dim
        
    def build(self, input_shape):
        self.pos_emb = self.add_weight(
            name='position_embedding',
            shape=(self.max_len, self.embed_dim),
            initializer='glorot_uniform',
            trainable=True)
        super().build(input_shape)
    
    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        positions = tf.cast(positions, tf.int32)
        return x + tf.nn.embedding_lookup(self.pos_emb, positions)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_len': self.max_len,
            'embed_dim': self.embed_dim
        })
        return config

class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim
        
    def build(self, input_shape):
        self.query = Dense(self.embed_dim)
        self.key = Dense(self.embed_dim)
        self.value = Dense(self.embed_dim)
        self.combine = Dense(self.embed_dim)
        super().build(input_shape)
    
    def call(self, x):
        batch_size = tf.shape(x)[0]
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        q = tf.reshape(q, (batch_size, -1, self.num_heads, self.head_dim))
        q = tf.transpose(q, [0, 2, 1, 3])
        
        k = tf.reshape(k, (batch_size, -1, self.num_heads, self.head_dim))
        k = tf.transpose(k, [0, 2, 1, 3])
        
        v = tf.reshape(v, (batch_size, -1, self.num_heads, self.head_dim))
        v = tf.transpose(v, [0, 2, 1, 3])
        
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention = tf.nn.softmax(matmul_qk / tf.math.sqrt(dk), axis=-1)
        output = tf.matmul(scaled_attention, v)
        
        output = tf.transpose(output, [0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.embed_dim))
        return self.combine(output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads
        })
        return config

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim  # 添加这行
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate  # 添加这行
        
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='swish'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=True):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'rate': self.rate
        })
        return config
def residual_conv_block(x, filters, kernel_size, strides, weight_decay, dropout_rate):  # 添加dropout_rate参数
    shortcut = Conv1D(filters, 1, strides=strides, padding='same',
                    kernel_regularizer=l2(weight_decay))(x)
    
    x = Conv1D(filters, kernel_size, strides=strides, padding='same',
              activation='swish', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(dropout_rate)(x)  # 使用传入的dropout_rate
    x = Conv1D(filters, kernel_size, padding='same',
              activation='swish', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    
    return Add()([x, shortcut])
# ==================== 模型构建 ====================
def create_ptm_model(config):
    # 序列输入分支（不变）
    seq_inputs = []
    for win_size in config.window_sizes:  
        seq_inputs.append(Input(shape=(win_size,),  name=f'seq_input_{win_size}'))
 
    # 结构特征输入（不变）
    struct_input = Input(shape=(config.struct_feature_dim,),  name='struct_input')
    esm_input = Input(shape=(config.esm_dim,),  name='esm_input')
 
    # 多尺度序列特征提取（修改部分）
    seq_branches = []
    for i, (win_size, inp) in enumerate(zip(config.window_sizes,  seq_inputs)):
        # 嵌入层和位置编码（不变）
        x = Embedding(config.max_features,  config.embed_dim)(inp) 
        x = PositionEmbedding(max_len=win_size, embed_dim=config.embed_dim)(x) 
        x = SpatialDropout1D(config.weight_decay)(x) 
        
        # 新增残差卷积块 
        x = residual_conv_block(
            x,
            filters=config.embed_dim, 
            kernel_size=3,
            strides=1,
            weight_decay=config.weight_decay, 
            dropout_rate=config.dropout_rate  # 添加这个参数
        )
        
        # 原有Transformer块（不变）
        x = TransformerBlock(config.embed_dim,  config.num_heads,  config.ff_dim)(x) 
        x = TransformerBlock(config.embed_dim,  config.num_heads,  config.ff_dim)(x) 
        
        # 注意力池化（不变）
        attention = Dense(1, activation='tanh')(x)
        attention = Flatten()(attention)
        attention = Activation('softmax')(attention)
        attention = RepeatVector(config.embed_dim)(attention) 
        attention = Permute([2, 1])(attention)
        x = Multiply()([x, attention])
        x = Lambda(lambda xin: K.sum(xin,  axis=1))(x)
        
        seq_branches.append(x) 
 
    # 后续结构特征和ESM特征处理（不变）
    struct_branch = Dense(128, activation='swish')(struct_input)
    struct_branch = BatchNormalization()(struct_branch)
    struct_branch = Dropout(config.dropout_rate)(struct_branch) 
    
    esm_branch = Dense(512, activation='swish')(esm_input)
    esm_branch = BatchNormalization()(esm_branch)
    esm_branch = Dropout(config.dropout_rate)(esm_branch) 
    esm_branch = Dense(256, activation='swish')(esm_branch)
    
    # 合并和输出层（不变）
    merged = Concatenate()(seq_branches + [struct_branch, esm_branch])
    merged = Dropout(config.dropout_rate)(merged) 
    
    x = Dense(512, activation='swish', kernel_regularizer=l2(config.weight_decay))(merged) 
    x = BatchNormalization()(x)
    x = Dropout(config.dropout_rate)(x) 
    
    x = Dense(256, activation='swish', kernel_regularizer=l2(config.weight_decay))(x) 
    x = BatchNormalization()(x)
    
    outputs = Dense(2, activation='softmax',
                   bias_initializer=tf.keras.initializers.Constant([0.1,  1.0]))(x)
    
    return Model(inputs=seq_inputs + [struct_input, esm_input], outputs=outputs)