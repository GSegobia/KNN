USERS_NUMBER = 943
ITENS_NUMBER = 1682

MIN_ITENS_RATED = 10

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

function global_mean(vector_a)
  mean_value = sum(vector_a)/length(find(x->(x > 0), vector_a))

  return round(mean_value)
end

function item_global_means(rates)
  items_means = []

  for i in 1:ITENS_NUMBER
    push!(items_means, global_mean(rates[:, i]))
  end

  return items_means
end

function min_items_number(min_number, vector_a, vector_b)
  vector_aux = vector_a .* vector_b

  number_items = length(find(x->(x > 0), vector_aux))

  if number_items >= min_number
    return true
  end

  return false
end

function similarity_table(rates_table, similarity_func, users_number)
  table = eye(users_number, users_number)

  for i in 1:users_number
    for j in i+1:users_number
      if min_items_number(MIN_ITENS_RATED, rates_table[i, :], rates_table[j, :])
        table[j, i] = 0
      else
        table[j, i] = similarity_func(rates_table[i, :], rates_table[j, :])
      end
    end
  end

  table = table + table'

  return table
end

function knn(K, vector)
  neighbors = []

  vector_aux = copy(vector)

  for i in 1:K
    index_change = find(r->r==maximum(vector_aux), vector_aux)
    vector_aux[index_change] = 0
    push!(neighbors, index_change)
  end

  return neighbors
end

function item_rated(user, similar_user, item, rates)
  if rates[user, item] > 0
    if rates[similar_user, item] > 0
      return true
    end
  end

  return false
end

global predictions = 0
global g_mean = 0
global predict_zero = 0
global g_mean_zero = 0

function user_based_prediction(K, rates, similarity, user, item, items_global_means)
  most_similar_list = knn(K+1, similarity[:, user])[2:end]

  rate = 0

  num = 0
  den = 0

  # println("\n\n")
  # println("-----START PREDICTION-----")

  max_neighbours = 3

  # println("Max Neighbours $(max_neighbours)")

  for i in most_similar_list
    if max_neighbours > 0
      if item_rated(user, i[1], item, rates)

        # println()
        # println("\[$(i[1])\] => Similarity: $(similarity[user, i[1]])")
        # println("\[$(i[1])\] => Item Rate: $(rates[i[1], item])")

        num += similarity[user, i[1]] * rates[i[1], item]
        den += similarity[user, i[1]]

        max_neighbours -= 1
        #
        # println()
        # println("Current Numerator $(num)")
        # println("Current Denominator $(den)")
        # println("Current Neighbours $(max_neighbours)")
      end
    end
  end

  if max_neighbours > 0
    rates = items_global_means[item]
    global g_mean += 1

    if rates == 0
      global g_mean_zero += 1
    end
  else
    rates = num/den
    global predictions += 1

    if rates == 0
      global predict_zero += 1
    end
  end

  # println("Item Global Mean $(items_global_means[item])")
  # println("Rate Predicted $(num/den)")
  #
  # println("-----END PREDICTION-----")
  # println("\n\n")

  return rates
end

function predict_test(rates_test, test, K, rates_training, similarity, items_global_means)
  (rows, columns) = size(rates_test)

  count = 0
  error = 0

  for i in test
    item = convert(Int64, round(i[1]/rows, RoundUp))
    user = i[1]%rows != 0 ? i[1]%rows : rows

    rates_test[user, item] = user_based_prediction(K, rates_training, similarity, user, item, items_global_means)

    count += 1

    if rates_test[user, item] == 0
      erro += 1
    end
  end
  println("Count: $(count)")
  println("Zeros: $(error)")
end

function set_training(users_rates)
  rates = find(r->r!=0, users_rates)

  training_rates = copy(users_rates)

  training = find(r->r, shuffle(1:length(rates)) .> (length(rates) * 0.2))
  test = setdiff(1:length(rates), training)

  for i in test
    training_rates[rates[i]] = 0
  end

  return training, test, training_rates
end

f = open(file_dir)

file_content = readdlm(f)

close(f)

# matrix_training = file_content[training, :]
# matrix_test = file_content[test, :]

#file_content = sortrows(file_content, by=x->(x[1],x[2]))

#users_rates_training = file_content[training, :]

# println(size(file_content))
# println(size(users_rates_training))
print("Time User Rates: ")
@time users_rates = rates_matrix(file_content, USERS_NUMBER, ITENS_NUMBER)
println()
print("Time User Rates Training: ")
@time (training, test, users_rates_training) = set_training(users_rates)
println()
print("Time User Rates Test: ")
@time users_rates_test = copy(users_rates_training)
println()
print("Time Similarity Table: ")
@time similarity = similarity_table(users_rates_training, cosine_similarity, USERS_NUMBER)
println()
print("Time Items Global Mean: ")
@time items_global_mean = item_global_means(users_rates_training)
println()
println("Count Global Mean Zero: $(length(find(r->r==0, items_global_mean)))")
println()
print("Time User Rates Predict: ")
@time predict_test(users_rates_test, test, 10, users_rates_training, similarity, items_global_mean)
println()
println("User Rates Final: $(length(find(r->r!=0, users_rates_test)))")
println()
println()
println("Global Mean Used: $(g_mean)")
println("Global Mean Zero: $(g_mean_zero)")
println("Predicted Rates: $(predictions)")
println("Predicted Zero: $(predict_zero)")

# predict_test(rates_test, test, K, rates_training, similarity, items_global_means)
#@time predict_test(users_rates, users_rates_training, test)
#writedlm("similarity.data", similarity)
#println(items_global_mean)
#println(length(similarity))
#println(whos())
