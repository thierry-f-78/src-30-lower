#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <windows.h>
#include <omp.h>

#define N 1000000
#define RUNS 1000

/*
 * LA STRUCTURE DE BASE :
 * Au lieu d'utiliser des `char*` de taille variable, on fige la taille à 16 octets.
 * C'est le prérequis architectural de notre hack : on remplace un problème
 * de dimension inconnue par une comparaison de blocs fixes.
 */
typedef struct {
    char data[16];
} string16;

volatile bool sink;

/* --- Générateur de données --- */
static uint64_t rng_state = 42;
static inline uint32_t fast_rand(void) {
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng_state >> 32);
}

static inline int rand_range(int min, int max) {
    return min + (fast_rand() % (max - min + 1));
}

static string16 random_word(void) {
    string16 s;
    /*
     * HACK (Partie 1) : Le Zero-Padding.
     * On initialise TOUS les octets à zéro (\0).
     * Si le mot généré fait 10 caractères, les 6 octets restants garantissent
     * un bruit nul. On n'aura pas d'octets "poubelles" à comparer par erreur.
     */
    memset(s.data, 0, 16);
    int len = rand_range(8, 15);
    for (int i = 0; i < len; i++) {
        s.data[i] = 'a' + rand_range(0, 25);
    }
    return s;
}

static string16 random_case(string16 src) {
    string16 dst = src;
    for (int i = 0; i < 16 && dst.data[i] != '\0'; i++) {
        if (rand_range(0, 1) == 0) {
            dst.data[i] -= ('a' - 'A'); // Alterne aléatoirement entre majuscule et minuscule
        }
    }
    return dst;
}

int main(void) {
    /*
     * ALIGNEMENT MÉMOIRE :
     * Un registre AVX-512 charge 64 octets (512 bits) d'un coup.
     * Pour utiliser l'instruction de chargement rapide `_mm512_load_si512` sans
     * déclencher de faute matérielle (SegFault), l'adresse de départ du tableau
     * doit impérativement être un multiple de 64.
     */
    string16* as = (string16*)_aligned_malloc(N * sizeof(string16), 64);
    string16* bs = (string16*)_aligned_malloc(N * sizeof(string16), 64);

    if (!as || !bs) return 1;

    // Remplissage des données de test
    for (int i = 0; i < N; i++) {
        string16 w = random_word();
        as[i] = random_case(w);
        if (rand_range(0, 1) == 0) {
            bs[i] = random_case(w);
        }
        else {
            bs[i] = random_case(random_word());
        }
    }

    // Préchauffage du pool de threads OpenMP pour ne pas fausser le premier chrono
#pragma omp parallel
    {
        (void)omp_get_thread_num();
    }

    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);

    /*
     * HACK (Partie 2) : Le Masque Bitwise Magique.
     * On prépare un registre AVX-512 rempli de la valeur 0x20 (32 en décimal)
     * sur chacun de ses 64 octets.
     * Pourquoi 0x20 ?
     * 1. Sur une lettre ASCII (ex: 'A' 0x41), un OU binaire avec 0x20 donne 'a' (0x61).
     * 2. Sur notre padding de zéros (0x00), un OU binaire avec 0x20 donne ' ' (0x20, l'espace).
     * Résultat : On n'a plus besoin de chercher la fin de la chaîne (\0).
     * Le padding de zéros devient un padding d'espaces stricts de part et d'autre,
     * et les majuscules deviennent des minuscules. Tout est uniformisé.
     */
    __m512i mask_low = _mm512_set1_epi8(0x20);

    printf("=== Benchmark AVX-512 + OpenMP (4 threads, %d passages) ===\n", RUNS);

    for (int run = 0; run < RUNS; run++) {
        int mt_matches = 0;
        int i;

        QueryPerformanceCounter(&start);

        // Découpage du travail sur 4 threads
#pragma omp parallel for reduction(+:mt_matches) schedule(static) num_threads(4)
        for (i = 0; i < N; i += 4) {
            /*
             * ÉTAPE 1 : Chargement
             * On charge 4 chaînes de 16 octets (soit 64 octets) en une seule instruction.
             */
            __m512i v1 = _mm512_load_si512((const __m512i*)&as[i]);
            __m512i v2 = _mm512_load_si512((const __m512i*)&bs[i]);

            /*
             * ÉTAPE 2 : Normalisation (Le fameux Hack)
             * On applique notre masque 0x20 sur les 64 octets simultanément.
             * Les lettres passent en minuscules, les \0 de padding deviennent des espaces.
             * Le tout sans AUCUN if, ni boucle, ni appel à strlen().
             */
            __m512i v1_low = _mm512_or_si512(v1, mask_low);
            __m512i v2_low = _mm512_or_si512(v2, mask_low);

            /*
             * ÉTAPE 3 : Comparaison Vectorielle
             * On compare les 64 octets de v1_low avec les 64 octets de v2_low.
             * L'instruction renvoie un entier de 64 bits (uint64_t).
             * Chaque bit à 1 dans ce uint64_t signifie "les octets à cette position sont égaux".
             */
            uint64_t cmp = _mm512_cmpeq_epi8_mask(v1_low, v2_low);

            /*
             * ÉTAPE 4 : Extraction des 4 résultats
             * Notre uint64_t contient le résultat pour 4 chaînes de 16 octets.
             * On découpe ce masque 64 bits en 4 blocs de 16 bits.
             * Si un bloc de 16 bits vaut 0xFFFF (tous les bits à 1), cela signifie que
             * les 16 octets de la chaîne correspondent parfaitement.
             * L'opérateur == convertit ce Vrai/Faux en 1 ou 0, qu'on additionne au total.
             */
            mt_matches += ((cmp & 0x000000000000FFFFULL) == 0x000000000000FFFFULL); // Chaîne 1
            mt_matches += ((cmp & 0x00000000FFFF0000ULL) == 0x00000000FFFF0000ULL); // Chaîne 2
            mt_matches += ((cmp & 0x0000FFFF00000000ULL) == 0x0000FFFF00000000ULL); // Chaîne 3
            mt_matches += ((cmp & 0xFFFF000000000000ULL) == 0xFFFF000000000000ULL); // Chaîne 4
        }

        QueryPerformanceCounter(&end);

        double elapsed_ms = ((double)(end.QuadPart - start.QuadPart) * 1000.0) / (double)freq.QuadPart;
        printf("Run %2d : %.3f ms (Matches: %d)\n", run + 1, elapsed_ms, mt_matches);

        sink = (mt_matches > 0);
    }

    _aligned_free(as);
    _aligned_free(bs);
    return 0;
}