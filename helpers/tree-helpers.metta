;; (AND A (AND A B))

;; Tree Definition using only tuples.
(Tree AND False () ((Tree A True) (Tree AND False ((Tree A True) (Tree B True)))))

;; Tree Definition using recursive list data structure
(TreeNode (Value Nil False AND) Nil (Cons (TreeNode (Value A True LITERAL) Nil Nil) (Cons (TreeNode (Value Nil False AND) Nil (Cons (TreeNode (Value A True LITERAL) Nil Nil) (Cons (TreeNode (Value B True LITERAL) Nil Nil) Nil))) Nil)))

;; N.D. Tree: Each branch of the tree needs to be kept so that the tree is fully represented.
(= (x) (Tree AND False Nil (Tree A True)))
(= (x) (Tree AND False Nil (Tree AND False Nil (Tree A True))))
(= (x) (Tree AND False Nil (Tree AND False Nil (Tree B True))))
!(collapse x)
(= (getBranches (Tree $type $constraint $guardSet $children)) $children)
;; !(getBranches (x))

;; N.D. Tree: With each children is stored along with their tree level.
(= (y) (Tree 1 AND False Nil (Tree 2 A True)))
(= (y) (Tree 1 AND False Nil (Tree 2 AND False)))
(= (y) (Tree 2 AND False Nil (Tree 3 A True)))
(= (y) (Tree 2 AND False Nil (Tree 3 B True)))

(= (getChildren $lvl $tree) (unify (Tree $lvl $value $constraint $guardSet $children) $tree $children (empty)))
(= (removeChild $child $tree)
    (let $tree (unify (Tree $lvl $value $constraint $gs $child) $tree (Tree $lvl $value $constraint $gs $child) ())
        (remove-atom &self (= (y) $tree))
    )
)
(=(buildTree $exp)
)


;; !(getChildren 1 (y))
;; !(removeChild (Tree 3 A True) (y))
;; !(removeChild (Tree 2 $v $c) (y))
;; !(getChildren 1 (y))
