---
title: Interfacing with Simulators for RL
date: 2017-05-14 10:20
modified: 2017-05-15 19:30
category: ReinforcementLeanring
Tags: RL, Python, SWIG, Authors
summary: These days, many people want to use one of many new libraries written in python to train deep learning models. In general, python has many powerful and easy to use libraries for performing machine learning. However, many applications that generate data are written in other languages. In particular, many simulators for RL applications are written in C++. Here I will focus on an interface I have found useful for wrapping physics-based simulators in C++.
author: Glen Berseth
---

# Intro

These days, many people want to use one of many new libraries written in python to train deep learning models. In general, python has many powerful and easy to use libraries for performing machine learning. However, many applications that generate data are written in other languages. In particular, many simulators for RL applications are written in C++. Here I will focus on an interface I have found useful for wrapping physics-based simulators in C++.

## The Interface

We want to able to support two types of control flow,
one for learning and one for simulation.
I find it helpful to have these two different methods as we often want to simulate things faster than we would typically render them.
Also, during rendering, we might want to interact with the simulation, for example, to give an agent in the simulation a little shove.

### Examples of The Flow of Control

For Rendering the simulation:
```
animate()
    // Might run the physics in the simulation faster than the frame rendering.
    For n = 1 .. num_substeps
        if simulation needs new action
            s <= get state()
            a <= get new action
            apply the action
            sim.updateAction(a)
    
        update simulation

    postredisplay()
```


For Training:
```
act(action)
    sim.updateAction(a)
    While (not needsNewAction())
        update sim()

simEpoch(actor, env)
    n=0
    tuples=[]
    While (not endOfEpisode() or n < 100)
        S = env.getState()
        A = policy(s)
        R  = env.act(a)
        tuples.append((S, A, R))
        n+=1

    return tuples
```

## The code

Being able to use the above pseudo code is made possible by implementing the following interface.

```
/*
 * SimulationWrapper.h
 *
 *  Created on: Dec 9, 2016
 *      author: Glen
 */

#include <vector>

class SimulationWrapper {
public:

    /// Create the wrapper given some configuration
	SimulationWrapper(Configuration * config);
	virtual ~SimulationWrapper();
    /// Get the Current observation the agent can 'see'
	virtual std::vector<double> getObservation();
    /// Has the simulation reached the end of an epoch/episode
	virtual bool endOfEpoch();
    /*
    Step the simulation through to the end of the action
    */
	virtual double act(std::vector<double> action);
    /*
    Update the current agent/controller parameters
    */
	virtual double updateAction(std::vector<double> action);
	/// Perform one simulation update
	virtual void update();
	/// check whether or not the last action has completed and a new action is needed
	virtual bool needUpdatedAction();
    /*
     Gets the state of the simuation not just the observation.
    This state should be detailed enough to set the entire state of the simulation back to this state.
    */
	virtual std::vector<double> getSimState();
    /*
        Sets the state of the simulation to the given state.
    */
	virtual void setSimState(std::vector<double> state_);
    /*
        Computes the observation from the given simulation state.
    */
	virtual std::vector<double> getObservationFromSimState(std::vector<double> state_);

    /// Initialize the agent in the simulation, prepares all datastructures to begin simulation
	virtual void init();
    /// clears the data for the old epoch and creates a new epoch to simulate
	virtual void initEpoch();
    /// Generates data for a new epoch
	virtual void generateEnvironmentSample();
    /// Gets a good measure to evaluate the performance of agent cross an entire epoch. For example average reward over epoch/episode
	virtual std::vector<double> getEvaluationData();

    /// unloads everything from the simulation in preparation for termination.
	virtual void finish();
    /// Clears the data for the current epoch/episode
	virtual void clear();

    /// Get a pointer to the current actor. Can help with computing reward function values
	virtual SimulationActor * getActor() const {return _actor;}
	/// Has the agent fallen (into a non recoverable state)
	virtual bool agentHasFallen();
    /// Calculate the current reward. This is often computed over the simulation since the beging of an action.
	virtual double calcReward() const ;

	/// Interactive functions to doing things in the simulation, like resetting and throwing objects at the character
	virtual void onKeyEvent(int key, int mouseX, int mouseY);

private:
	SimulationAgent * _actor;

};


```


