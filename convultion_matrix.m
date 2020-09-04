function  f_output = convultion_matrix(A,B)
f=zeros(28,28);
f2=zeros(24,24);
    for i=3:26
        for j=3:26
            for u=1:5
                for v=1:5
          f(i,j)=f(i,j)+A(i+3-u,j+3-v)*B(u,v);
                end
            end
        end
    end
    f2(1:24,1:24)=f(3:26,3:26);
f_output = f2;
end