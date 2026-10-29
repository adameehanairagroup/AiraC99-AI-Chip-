/**
 * ============================================================================
 * AIRA CORE - SOVEREIGN EDGE AI INFERENCE ENGINE
 * Main Transformer Model Architecture & Execution API
 * 
 * Target Architecture: LLaMA / Mistral / TinyLlama / Sovereign Decoder
 * 
 * Architect: Er. Adam Eehan (Founder & CEO, Aira Group of Technology)
 * Co-Pilot: Aira 3 Ultra (Engine v3.6)
 * ============================================================================
 */

#ifndef AIRA_CORE_H
#define AIRA_CORE_H

#include "hal/hal_os.h"
#include "aira_math.h"
#include "aira_tokenizer.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AIRA_CORE_VERSION "1.0.0-SOVEREIGN"
#define AIRA_MAGIC_HEADER 0x61697261 /* 'aira' in ASCII */

/* Hyperparameters matching transformer configuration */
typedef struct {
    int dim;        /* Transformer embedding dimension (e.g. 288, 512, 2048) */
    int hidden_dim; /* FFN hidden dimension (e.g. 768, 1408, 5632) */
    int n_layers;   /* Number of transformer layers */
    int n_heads;    /* Number of query attention heads */
    int n_kv_heads; /* Number of key/value heads (Grouped Query Attention) */
    int vocab_size; /* Vocabulary size */
    int seq_len;    /* Maximum sequence length (context window) */
} aira_config_t;

/* Transformer Weights */
typedef struct {
    float* token_embedding_table; /* (vocab_size, dim) */
    float* rms_att_weight;        /* (n_layers, dim) */
    float* wq;                    /* (n_layers, dim, n_heads * head_size) */
    float* wk;                    /* (n_layers, dim, n_kv_heads * head_size) */
    float* wv;                    /* (n_layers, dim, n_kv_heads * head_size) */
    float* wo;                    /* (n_layers, n_heads * head_size, dim) */
    float* rms_ffn_weight;        /* (n_layers, dim) */
    float* w1;                    /* (n_layers, dim, hidden_dim) - gate_proj */
    float* w2;                    /* (n_layers, hidden_dim, dim) - down_proj */
    float* w3;                    /* (n_layers, dim, hidden_dim) - up_proj */
    float* rms_final_weight;      /* (dim) */
    float* wcls;                  /* (vocab_size, dim) - classifier head (tied or untied) */
} aira_weights_t;

/* Dynamic Activation Buffers & KV Cache */
typedef struct {
    float* x;           /* Activation at current time step (dim) */
    float* xb;          /* Same, inside a residual branch (dim) */
    float* xb2;         /* Additional branch buffer (dim) */
    float* hb;          /* Buffer for hidden dimension in FFN (hidden_dim) */
    float* hb2;         /* Buffer for hidden dimension in FFN (hidden_dim) */
    float* q;           /* Query vectors (dim) */
    float* k;           /* Key vector (n_kv_heads * head_size) */
    float* v;           /* Value vector (n_kv_heads * head_size) */
    float* att;         /* Buffer for attention scores (n_heads, seq_len) */
    float* logits;      /* Output logits over vocabulary (vocab_size) */
    float* key_cache;   /* KV Cache for keys: (n_layers, seq_len, n_kv_heads * head_size) */
    float* value_cache; /* KV Cache for values: (n_layers, seq_len, n_kv_heads * head_size) */
} aira_runstate_t;

/* Sampler Settings */
typedef struct {
    float    temperature;
    float    topp;
    uint64_t rng_state;
} aira_sampler_t;

/* Complete Transformer Instance */
typedef struct {
    aira_config_t    config;
    aira_weights_t   weights;
    aira_runstate_t  state;
    hal_mmap_t       model_map;
    bool             is_mapped;
} aira_transformer_t;

/* --- Lifecycle & Execution API --- */
int  aira_transformer_load(aira_transformer_t* model, const char* checkpoint_path);
void aira_transformer_free(aira_transformer_t* model);

/* Perform a forward pass for a single token at position 'pos' */
float* aira_forward(aira_transformer_t* model, int token, int pos);

/* Sampler Initialization & Sampling */
void aira_sampler_init(aira_sampler_t* sampler, float temperature, float topp, uint64_t rng_seed);
int  aira_sample(aira_sampler_t* sampler, float* logits, int vocab_size);

/* Complete Text Generation Loop */
void aira_generate(aira_transformer_t* model, 
                   aira_tokenizer_t* tokenizer, 
                   aira_sampler_t* sampler, 
                   const char* prompt, 
                   int steps);

#ifdef __cplusplus
}
#endif

#endif /* AIRA_CORE_H */
