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
;;a function that binarizes an n-ary expression
;; (= (binarize $expression $parentNode $accum)
;;     (if (== $expression ())
;;         $accum
;;         (let* (
;;                 (($head $tail) (decons-atom $expression))
;;                 ($isHeadJunctor (or (== $head AND) (== $head OR))) ;;isHeadJunctor
;;             )
;;             (if $isHeadJunctor
;;                 (if )

;;             )
;;         )
;;     )
;; )
(=(kolmogorov $exp)
    (if (== () $exp)
        0
        (let ($head $tail) (decons-atom $exp)
            (if (== (get-metatype $head) Expression)
                (+ (kolmogorov $head) (kolmogorov $tail))
                (+ 1 (kolmogorov $tail))
            )
        )
    )
     
)
;;This function counts symbols if they are not literals in an expression
(= (countLiterals $expression)
    (if (== $expression ())
        0
        (let* (
                (($head $tail) (decons-atom $expression))
                ($isHeadJunctor (or (== $head AND) (== $head OR))) ;;isHeadJunctor
            )
            (if (and (== (get-metatype $head) Symbol) (not $isHeadJunctor))

                (+ 1 (countLiterals $tail))
                (countLiterals $tail)
            )
        )
    )
)
(= (foldr $func $value $list)
    (if (== $list ())
        $value
        (foldr $func ($func $value (car-atom $list)) (cdr-atom $list))
    )
)
;;!(countLiterals (AND A B (OR C D) E))
;; !(kolmogorov (AND A B (OR C D) E))
!(foldr * 1 (1 2 3 4 5))

;;!(decons-atom (1 2 3 4 5))
;;!(tree-depth (AND x1 (OR x1 (AND x3 x4) x5)))
