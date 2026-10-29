/**
 * ============================================================================
 * AIRA CORE - TOKENIZER & VOCABULARY ENGINE IMPLEMENTATION
 * 
 * Supports BPE & SentencePiece vocabularies.
 * Converts token IDs into human-readable text and handles raw byte fallback.
 * 
 * Architect: Er. Adam Eehan (Aira Group of Technology)
 * Co-Pilot: Aira 3 Ultra (Sovereign Engine v3.6)
 * ============================================================================
 */

#include "aira_tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int aira_tokenizer_init(aira_tokenizer_t* t, const char* tokenizer_path, int vocab_size) {
    if (!t || !tokenizer_path) return -1;
    memset(t, 0, sizeof(aira_tokenizer_t));
    t->vocab_size = vocab_size;

    /* Initialize byte pieces cache */
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE* file = fopen(tokenizer_path, "rb");
    if (!file) {
        return -1;
    }

    if (fread(&t->max_token_length, sizeof(uint32_t), 1, file) != 1) {
        fclose(file);
        return -1;
    }

    t->vocab = (char**)malloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
    if (!t->vocab || !t->vocab_scores) {
        if (t->vocab) free(t->vocab);
        if (t->vocab_scores) free(t->vocab_scores);
        fclose(file);
        return -1;
    }

    for (int i = 0; i < vocab_size; i++) {
        if (fread(&t->vocab_scores[i], sizeof(float), 1, file) != 1) break;
        int32_t len = 0;
        if (fread(&len, sizeof(int32_t), 1, file) != 1) break;

        t->vocab[i] = (char*)malloc((size_t)len + 1);
        if (fread(t->vocab[i], 1, (size_t)len, file) != (size_t)len) {
            t->vocab[i][0] = '\0';
        } else {
            t->vocab[i][len] = '\0';
        }
    }

    fclose(file);
    return 0;
}

const char* aira_tokenizer_decode(aira_tokenizer_t* t, int prev_token, int token) {
    (void)prev_token;
    if (!t || !t->vocab || token < 0 || token >= t->vocab_size) {
        return "";
    }
    const char* piece = t->vocab[token];
    if (!piece) return "";

    /* Handle raw byte tokens: '<0xXX>' */
    if (piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>') {
        unsigned int byte_val = 0;
        if (sscanf(piece + 3, "%2x", &byte_val) == 1) {
            return (const char*)&t->byte_pieces[(byte_val & 0xFF) * 2];
        }
    }
    return piece;
}

void aira_tokenizer_free(aira_tokenizer_t* t) {
    if (!t) return;
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) {
            if (t->vocab[i]) free(t->vocab[i]);
        }
        free(t->vocab);
        t->vocab = NULL;
    }
    if (t->vocab_scores) {
        free(t->vocab_scores);
        t->vocab_scores = NULL;
    }
}
