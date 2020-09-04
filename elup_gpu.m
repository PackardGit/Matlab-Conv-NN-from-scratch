function fr = elu2(x)
    f = zeros(length(x),1);
    for i = 1:length(x)

        f(i) = (1-exp(-x(i)))/(1+exp(-x(i)));

    end
    fr = f;
end