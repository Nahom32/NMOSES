(= (exec True) True)
(= (exec False) False)

(= (exec ($op $exp1 $exp2))
    (if (== $op AND)
        (and (exec $exp1) (exec $exp2))
        (if (== $op OR)
            (or (exec $exp1) (exec $exp2))
            ERROR
        )
    )
)
(= (naryExec $exp)
    (let* (
        (expType (get-metatype $exp))
        (isExpression (== expType Expresion))
        
    )
        (if isExpression
           (let* (
                    (head (car-atom $exp))
                    (tail (cdr-atom $exp))
                    (isHeadJunctor (or (== head OR) (== head AND)))
                )
                (if isHeadJunctor
                    ()
                )
           )
        )
    )
)
(=(backward-chainer $data $space)
    ()
)