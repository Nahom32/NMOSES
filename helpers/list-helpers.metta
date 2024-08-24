(list x 1)
(list x 2)
(list x 3)

(list y 2)
(list y 3)
(list y 4)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(= (append $list-name $x)(
    add-atom &self (list $list-name $x)
))
;; !(append x 9) 

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(=(return-list $list-name)(
    match &self (list $list-name $y) $y
))
;; !(return-list x)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(=(search $list-name $x)(
   match &self (list $list-name $x) $x
))
;; !(search x 3)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(=(intersect $list-name1 $list-name2 $intersectionList)(
    match &self (list $list-name1 $x)(
        match &self (list $list-name2 $y) (
            if (== $x $y) (add-atom &self (list w $x)) (empty)
            )
    )
))

;; !(intersect x y w)
;; !(return-list w)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(= (find-and-remove $list-name $val)(
    remove-atom &self (list $list-name $val)
))
;; !(find-and-remove x 2)
;; !(return-list x)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(= (find-and-replace $list-name $val $newVal)(
    (find-and-remove $list-name $val)
    (append $list-name $newVal)
))
;; !(find-and-replace x 9 8)
;; !(return-list x)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(= (listUnion $list-name1 $list-name2 $merged-list)(
    (listUnion-helper $list-name1 $merged-list)
    (listUnion-helper $list-name2 $merged-list)
))
(= (listUnion-helper $list-name $merged-list)(
    match &self (list $list-name $x) (
        add-atom &self (list $merged-list $x)
    )
))

;; !(listUnion x y z)
;; !(return-list z)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(= (double $x) (+ $x $x))
(= (map $func $list-name)(
    match &self (list $list-name $x) ($func $x)
))
;; !(map double x)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(= (exists-in-list $x $list-name)(
    match &self (list $list-name $y)(
        if (== $x $y) True (empty)
    )
))

;; !(exists-in-list 3 x)
