function f_output = activate_function1_derivative(A)
    [x, y, z]=size(A);
    f=zeros(x,y,z);
    for i=1:x
        for j=1:y
            for k=1:z
                f(i,j,k)=A(i,j,k)^2*(1-A(i,j,k)); %dc2/dk2 * dc2
            end
        end
    end
f_output = f;
end