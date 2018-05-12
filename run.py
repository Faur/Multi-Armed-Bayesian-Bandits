import numpy as np
import matplotlib.pyplot as plt
import sys

from multiprocessing import Pool

from bandit import *
from agents import *

def make_agents(env, k, max_steps):
	agents = [OptimalAgent(env), RandomAgent(k), FreqUCB(k), BayesUCB(k, max_steps), HierarchicalBayesUCB(k, max_steps), ThompsonSampling(k, max_steps), HierarchicalThompsonSampling(k, max_steps)]
	return agents

agents_names = ['Optimal', 'Random', 'FreqUCB', 'BayesUCB', 'HierarchicalBayesUCB', 'ThompsonSampling', 'HierarchicalThompsonSampling']


def run(args):
	k, max_steps, alpha, beta = args 
	env = KBandit(k, max_steps=max_steps, alpha=alpha, beta=beta)
	agents = make_agents(env, k, max_steps)

	rewards = []
	_, _, d, _ = env.reset()
	for a in agents:
		a.reset()
	while not d:
		draw = True
		rewards_ = []
		for agent in agents:
			a = agent.action()
			_, r, d, _ = env.step(a, draw)
			draw = False
			agent.update(a, r)
			rewards_.append(r)
		rewards.append(rewards_)

	return np.array(rewards).T

def runs(num_runs, args, mp=False):
	try:
		if mp:
			print('multiprocessing')
			proc_pool = Pool(num_runs)
			rewards = proc_pool.map(run, [args for i in range(num_runs)])
		else:
			print('single thread')
			rewards = []
			for i in range(num_runs):
				print('\rrun', i+1, 'of', num_runs, end=''); sys.stdout.flush()
				rewards.append(run(args))
			print()
	except KeyboardInterrupt:
		pass

	rewards = np.array(rewards)
	# rewards = np.mean(rewards, 0)
	return rewards


def plot_cumsum(rewards, agents_names, use_std=False):
	colors=plt.cm.rainbow(np.linspace(0,1,rewards.shape[1]))
	plt.figure(figsize=(6,6))
	for i in range(len(agents_names)-1):
	#	 plt.plot(np.cumsum(rewards[0] - rewards[i+1]), label=agents_names[i+1]+' regret')
		c = colors[i]
		cumsums = np.cumsum(rewards[:,0,:] - rewards[:,i+1,:], -1)
		means = np.mean(cumsums, 0)
		stds = np.std(cumsums, 0)
		plt.plot(means, c=c, lw=2, label=agents_names[i+1], alpha=0.75)
		if use_std:
			plt.plot(means+stds, c=c, linestyle='--', alpha=0.5)
			plt.plot(means-stds, c=c, linestyle='--', alpha=0.5)

	plt.legend()


def main():
	show = True
	save_rewards = False

	max_steps = 5000
	num_episodes = 25

	ks = [2, 20, 200]
	ks = [2, 20]

	if 0:
		print('SPECIAL RUN')
		ks = [2]
		max_steps = 100
		num_episodes = 1
		show = False
		save_rewards = False

	#########################################
	print('save_rewards', save_rewards)
	print('max_steps', max_steps)
	print('num_episodes', num_episodes)
	print('ks', ks)
	print()

	for k in ks:
		print('k =',k)
		alpha = 10; beta = 40
		args = (k, max_steps, alpha, beta)
		# env = KBandit(k, max_steps=max_steps, alpha=10, beta=40)
		# agents = make_agents(env, k, max_steps)

		rewards = runs(num_episodes, args, mp=True)
		if save_rewards:
			np.save('raw_rewards_'+str(k)+'.npy', rewards)

		plot_cumsum(rewards, agents_names)
		plt.title('Regret, k='+str(k))
		plt.draw()

		plot_cumsum(rewards, agents_names, True)
		plt.title('Regret, k='+str(k))
		plt.draw()
		print()

	if show:
		plt.show()

	print("Done")


	### VISUALIZE
	# plt.figure()
	# for i in range(len(agents)):
	#	 plt.plot(rewards[i], label=agents_names[i])
	#	 plt.title("Mean over " + str(num_episodes) + ' episodes')
	# plt.legend()
	# plt.draw()

	# plt.figure()
	# for i in range(len(agents)-1):
	#	 plt.plot(np.cumsum(rewards[0] - rewards[i+1]), label=agents_names[i+1]+' regret')
	# plt.legend()
	# plt.draw()
		
	# plt.figure()
	# for i in range(len(agents)-1):
	#	 one_step_diff = np.cumsum(rewards[0] - rewards[i+1])[1:] - np.cumsum(rewards[0] - rewards[i+1])[:-1]
	#	 plt.plot(one_step_diff, label=agents_names[i+1]+' regret, diff')
	# plt.legend()
	# plt.draw()
	# plt.show()

if __name__ == '__main__':
	main()