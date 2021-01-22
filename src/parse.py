def clean_features(array, order):
    """
    Accepts array of raw feature contributions
    Returns clean dict of contributions where variables are summed by category
    """
    sums = {i: 0 for i in order}

    for tup in array:
        feature, contribution = tup
        onehot = feature.find("onehotencoder")  # returns -1 if substr not in str
        if onehot != -1:
            index = int(feature[16])
            sums[order[index]] += contribution
        else:
            sums[feature] = contribution
    return sums


def order_features(bias, features):
    """
    Accepts a clean dictionary of features
    Returns a user-ready array of tuples (ordered) explaining feature contributions.
    1. Average n-bed m-bath apartment in NYC:
    2. remaining features...
    """
    ordered_features = []
    ordered_features.append(
        ("base", bias + features["bedrooms"] + features["bathrooms"])
    )
    for k, v in features.items():
        if k not in ["bedrooms", "bathrooms"]:
            ordered_features.append((k, v))
    return ordered_features


if __name__ == "__main__":
    order = ["housing_type", "laundry", "parking"]
    bias = 2000
    case = [
        ("onehotencoder__x0_apartment", 3.51469500233134),
        ("onehotencoder__x0_condo", -3.169949895013363),
        ("onehotencoder__x0_cottage/cabin", -0.01801470588235367),
        ("onehotencoder__x0_house", 1.2538370508949601),
        ("onehotencoder__x0_loft", -34.75664280098565),
        ("onehotencoder__x0_townhouse", 0.032218651319990355),
        ("onehotencoder__x1_laundry in bldg", 3.2291729452756703),
        ("onehotencoder__x1_laundry on site", -3.139606114765313),
        ("onehotencoder__x1_no laundry on site", 4.325598287300099),
        ("onehotencoder__x1_w/d hookups", -0.4149117911245594),
        ("onehotencoder__x1_w/d in unit", 269.0601755655627),
        ("onehotencoder__x2_attached garage", 44.95329389531284),
        ("onehotencoder__x2_detached garage", -0.36738519680435283),
        ("onehotencoder__x2_no parking", -4.146657261934127),
        ("onehotencoder__x2_off-street parking", 140.98904560783848),
        ("onehotencoder__x2_street parking", -112.40978319840299),
        ("onehotencoder__x2_valet parking", 0.7811362479112653),
        ("cats_ok", 47.555039089370574),
        ("dogs_ok", 37.9285369922712),
        ("bedrooms", -178.9682546451532),
        ("bathrooms", -178.96706524438642),
        ("no_smoking", 148.31449290409134),
        ("is_furnished", 10.895725356005993),
        ("wheelchair_acccess", 79.34635311635243),
        ("ev_charging", 3.051202277491352),
    ]

    clean = clean_features(case, order)
    print(order_features(bias, clean))