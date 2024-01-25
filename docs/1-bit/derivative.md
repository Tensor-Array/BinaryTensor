Before read this, you need to read:<br>
[operator](operator.md)

## Derivative rules

Here are some derivative function that only using one bit and bitwise operator.

| Common Functions | Function | Derivative |
|:-|:-:|:-:|
| Constant | c | 0 |
| Line | x<br>a & x | 1<br>a |
| Square | x & x | x ^ x |
| NOT | ~ x | 1 |

Here are some derivative rules that only using one bit and bitwise operator.

| Rules | Function | Derivative |
|:-|:-:|:-:|
| AND by constant | c & f(x) | c & f’(x) |
| AND rule | f(x) & g(x) | f’(x) & g’(x) |
| XOR rule | f(x) ^ g(x) | (f’(x) & g(x)) ^ (f(x) & g’(x)) |
| NOT rule | ~ f(x) | f’(x) |
| Chain Rule | f(g(x))| 	f’(g(x)) ^ g’(x) |
