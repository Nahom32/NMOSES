;; ! (register-module! ../../EDA)
;; ! (import! &self EDA:scoring:general-helpers)
;; ! (bind! &space (new-space))
;! (import! &test testData)


(= (addToSpace $row $space) (add-atom $space $row))

(= (removeFromSpace $row $space) (remove-atom $space $row))

(: foldr (-> (-> $a $b $b) $b $c $d))
(= (foldr $f $i $xs)
   (if (== $xs ())
       $i
       (let* (($h (car-atom $xs))
              ($t (cdr-atom $xs))
              ($ft (foldr $f $i $t)))
         ($f $h $ft))))

; replace expersion with Truth value from the space
(= (replaceWithTruth $expr $space $spaceContent)
    (if (== (get-metatype $expr) Expression) 
        (let* (
            ($op (car-atom $expr))
            ($tuple (cdr-atom $expr)) 
            ($substitutedArgs (collapse (replaceWithTruth (superpose $tuple) $space $spaceContent)) )

            ($tupleWithOp (cons-atom $op $substitutedArgs))
        )
            $tupleWithOp
        )
         (unify $space ($expr $value) $value 'undefined') 
    )
)

;a function to evaluateuate the boolean expression
(= (evaluate $expr)
    (if (== (get-metatype $expr) Grounded) ; T/ F
        $expr
        (let* (
            ($op (car-atom $expr))
            ($tuple (cdr-atom $expr))
        )
            (if (== $op AND)
                (evaluateAnd $tuple)
                (if (== $op OR)
                    (evaluateOr $tuple)
                    (if (== $op NOT)
                        (not (evaluate $tuple))
                        ERROR
                    )
                )
            )
        )
    )
)

(= (evaluateOr $args)
    (if (== $args ())
        False
        (let* (
            ($first (car-atom $args))
            ($rest (cdr-atom $args))
        )
            (if (== (evaluate $first) True)
                True
                (evaluateOr $rest)
            )
        )
    )
)

(= (evaluateAnd $args)
    (if (== $args ())
        True
        (let* (
            ($first (car-atom $args))
            ($rest (cdr-atom $args))
        )
            (if (== (evaluate $first) False)
                False
                (evaluateAnd $rest)
            )
        )
    )
)

;; (= (length $tuple)
;;     (if (== $tuple ())
;;         0
;;         (+ 1 (length (cdr-atom $tuple)))
;;     )
;; )

(= (counter $expr)
    (if (== $expr ())
        0 
        (let* (
            ($head (car-atom $expr)) 
            ($tail (cdr-atom $expr)) 
        )
            (if (== (get-metatype $head) Expression) 
                (+ (counter $head) (counter $tail))
                (if (or (== $head AND) (== $head OR)) 
                    (+ 1 (counter $tail)) 
                    (counter $tail) 
                )
            )
        )
    )
)








; measure the accuracy of the exprssion based on the given data 
(= (fitness $expr $data)
    (if (== $data ())
        0
        (let*(
            ($result (collapse (rowFitness $expr (superpose $data)))))

            (foldr + 0 $result)
        )
    )
)

; evaluate the fitness of a single row 
(= (rowFitness $expr $row )
    (let* (
            ($target (car-atom $row))
            ($inputs (cdr-atom $row))
            ($add (collapse (addToSpace (superpose $inputs) &space)))
            ($replacedExpr (replaceWithTruth $expr &space (collapse (match &space $x $x))))
            ($evaluateuatedResult (evaluate $replacedExpr))
            ($remove (collapse (removeFromSpace (superpose $inputs) &space)))
        )
        (if (== $evaluateuatedResult $target) 1 0)
    )
)



(= (penalizedFitness $expr $data $lambda)
    (let* (
        ($accuracy (fitness $expr $data)) 
        ($complexity (counter $expr))    
    )
        (- $accuracy (* $lambda $complexity))  
    )
)

;; Test Cases 1: Simple AND Expression
; !(penalizedFitness (AND A (OR B C)) 
;                    ((True (A True) (B True) (C False)) 
;                     (False (A False) (B True) (C True))) 
;                    0.1)

; ; Test Case 2: Nested OR Expression
; !(penalizedFitness (OR (AND A B) C) 
;                    ((True (A True) (B False) (C True)) 
;                     (True (A True) (B True) (C False)) 
;                     (False (A False) (B False) (C False))) 
;                    0.1) ; Expected: 2.8

; ; Test Case 3: Complex Expression
; !(penalizedFitness (OR (AND A (NOT B)) (AND C D)) 
;                    ((True (A True) (B False) (C True) (D True)) 
;                     (False (A False) (B True) (C False) (D False)) 
;                     (True (A True) (B False) (C False) (D True))) 
;                    0.1) ; Expected: 2.7



; !(fitness (OR (AND A (NOT B)) (AND C D))
;             ((True (A True) (B False) (C True) (D True)) 
;             (False (A False) (B True) (C False) (D False)) 
;             (True (A True) (B False) (C False) (D True))) )


; !(fitness (AND A (OR B C)) ( (True (A True) (B True) (C True)) (True (A False) (B False) (C True)) ) )

;;!(fitness (AND A (OR B C)) ( (True (A True) (B True) (C True)) (True (A False) (B False) (C True)) (False (A False) (B False) (C False)) ) )
;;!(fitness (AND A (OR B C)) ( (False (A False) (B True) (C True)) (False (A True) (B False) (C True)) ) )








