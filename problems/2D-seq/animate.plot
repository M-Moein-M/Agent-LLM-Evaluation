set term png size 1024,768 enhanced font ,12
set datafile separator whitespace
set grid
set hidden3d
set xrange [0:40]
set yrange [0:40]
set zrange [-2:2]

input(n) = sprintf("p-%d.dat", n)
output(n) = sprintf("%03d.png", n)

do for [i=1:50] {
    set output output(i)
  splot input(i) matrix using 1:2:3 with lines
}
