function fr = elup2(x)
    f = zeros(length(x),1);
    for i = 1:length(x)

        f(i) = (2*exp(x(i)))/((exp(x(i))+1)^2);
    end

    fr = f;
end