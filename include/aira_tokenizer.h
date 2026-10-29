/**
 * ============================================================================
 * AIRA CORE - TOKENIZER & VOCABULARY ENGINE
 * 
 * Supports BPE (Byte Pair Encoding) & SentencePiece tokenization.
 * Converts token IDs to UTF-8 text strings and handles byte fallback.
 * 
 * Architect: Er. Adam Eehan (Aira Group of Technology)
 * ============================================================================
 */

#ifndef AIRA_TOKENIZER_H
#define AIRA_TOKENIZER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char**   vocab;            /* Array of token string representations */
    float*   vocab_scores;     /* Scores for each token */
    int      vocab_size;       /* Number of tokens in vocabulary */
    uint32_t max_token_length; /* Longest token in vocabulary */
    unsigned char byte_pieces[512]; /* Cache for single byte fallback tokens */
} aira_tokenizer_t;

/* Initialize tokenizer from a binary tokenizer file */
int  aira_tokenizer_init(aira_tokenizer_t* t, const char* tokenizer_path, int vocab_size);

/* Decode token ID to string */
const char* aira_tokenizer_decode(aira_tokenizer_t* t, int prev_token, int token);

/* Free tokenizer resources */
void aira_tokenizer_free(aira_tokenizer_t* t);

#ifdef __cplusplus
}
#endif

#endif /* AIRA_TOKENIZER_H */
