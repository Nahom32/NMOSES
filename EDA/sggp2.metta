;; <start> -> <expr>
;; <expr>  -> <bop> <expr> <expr> |
;;            <uop> <expr> |
;;            <term>
;; <bop>   -> AND | OR
;; <uop>   -> NOT
!(bind! &grammarSpace (new-space))
!(bind! &stochasticGrammarSpace (new-space))
!(bind! &alphabetSpace (new-space))

(= (initializeGrammarSpace $grammarSpace)
    (add-reduct &grammar (superpose (
        (start expr)
        (expr (bop expr expr))
        (expr (uop expr))
        (expr term)
        (bop AND)
        (bop OR)
        (uop NOT)

    )
  )
)

)

(= (addListToSpace $space $list)
    (add-reduct $space (superpose $list))
)
!(get-atoms $space)

