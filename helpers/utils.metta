(= (reverse $list $accum)
    (if (== $list ())
        $accum
        (let* (
            ($head (car-atom $list))
            ($tail (cdr-atom $list))
        )
            (reverse $tail (cons-atom $head $accum))
        )
    )
)

(= (tree-depth $exp)
    (if (== (get-metatype $exp) Expression)
        (max (collapse (+ 1 (tree-depth (superpose $exp)))))
        
        1
    )
)
(= (max-helper $exp $val)
    (if (== $exp ())
        $val
        (let* (
            ($head (car-atom $exp))
            ($tail (cdr-atom $exp))

        )
            (if (> $head $val)
                (max-helper $tail $head)
                (max-helper $tail $val)
            )
        )
    )

)
(=(max $exp)
    (max-helper (cdr-atom $exp) (car-atom $exp))
)

;; (= (count-terminals $expression)
;;     (if  (not (== (get-metatype $expression) Expression))
;;         (+ 1 (count-terminals (superpose $expression)))
;;     )
;; )
(=(decons-atom $atom)
    ((car-atom $atom) (cdr-atom $atom))
)
!(decons-atom (1 2 3 4 5))
;;!(tree-depth (AND x1 (OR x1 (AND x3 x4) x5)))
