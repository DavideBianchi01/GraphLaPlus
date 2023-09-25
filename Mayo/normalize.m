function image = normalize(image)
image = (image - min(image(:))) / (max(image(:)) - min(image(:)));
end