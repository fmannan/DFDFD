function y = WeightFn(x, a, b, k1, k2)
% a= .25, b = .1, k1 = -.1, k2 = 10
% y = WeightFn(x, .25, .09, -.1, 10)
y = (a * abs(x)).^k1 + (b * abs(x)).^k2;

y(isinf(y)) = 0;
