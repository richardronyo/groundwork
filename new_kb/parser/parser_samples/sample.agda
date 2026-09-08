module Sample where

-- A simple inductive natural number type
data Nat : Set where
  zero : Nat
  suc  : Nat -> Nat

-- add two natural numbers
add : Nat -> Nat -> Nat
add zero    n = n
add (suc m) n = suc (add m n)

record Point : Set where
  field
    x : Nat
    y : Nat
