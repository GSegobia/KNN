USERS_NUMBER = 943
ITENS_NUMBER = 1682

file_dir = "ml-100k/u.data"

function rates_matrix(file_content, users_number, itens_number)
  complete_users_rates = zeros(users_number, itens_number)

  for i in 1:users_number
    user = find(x->(x == i), file_content[:, 1])

    for j in 1:length(user)
      complete_users_rates[i, convert(Int64, file_content[user[j], 2])] = file_content[user[j], 3]
    end
  end

  return complete_users_rates
end

function cosine_similarity(vector_a, vector_b)
  similarity = sum(vector_a .* vector_b)/sqrt(sum(vector_a .^ 2) * sum(vector_b .^ 2))

  return similarity
end

function similarity_table(rates_table, similarity_func, users_number)
  table = eye(users_number, users_number)

  for i in 1:users_number
    for j in i+1:users_number
      table[j, i] = similarity_func(rates_table[i, :], rates_table[j, :])
    end
  end

  return table
end

f = open(file_dir)

file_content = readdlm(f)

close(f)

#file_content = sortrows(file_content, by=x->(x[1],x[2]))
users_rates = rates_matrix(file_content, USERS_NUMBER, ITENS_NUMBER)
similarity = similarity_table(users_rates, cosine_similarity, USERS_NUMBER)

writedlm("similarity.data", similarity)
println(length(similarity))
println(whos())
