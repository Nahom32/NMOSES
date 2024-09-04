;;!(register-module )
!(import &self data-set-mapper)

!(mapper (OR (AND A (OR A B)) B) (Cons (A True) (Cons (B False) Nil)))