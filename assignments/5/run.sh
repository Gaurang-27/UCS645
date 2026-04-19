for i in 1 2 4 6 8; do
  mpicxx -O2 -o q1 q1.cpp && mpirun -np $i ./q1
  echo "------------------------"
done