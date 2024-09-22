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