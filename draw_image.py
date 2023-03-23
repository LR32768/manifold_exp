from image_generator import ImageGenerator

man_dim=20
seed=2333

g_list = ['/home/raylu/afhqcat.pkl', '/home/raylu/afhqdog.pkl']
gen = ImageGenerator(g_list, man_dim=man_dim, device=0, seed=seed)

#gen.navigate_manifold(out_dir='vis_2d_manifold')
gen.generate_dataset('./tmp', num_imgs=10)