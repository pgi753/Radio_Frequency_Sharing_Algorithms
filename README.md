# Radio_Frequency_Sharing_Algorithms

## DES-based Communication simulator

<img src="https://user-images.githubusercontent.com/65005179/176996743-a8ebabfb-8315-486f-9947-88c2e1b8186f.gif" width="80%" height="80%"/>

## Installation
Dependencies:
* [Pytorch](https://pytorch.org/) : 1.9.1
* [tensorflow](https://www.tensorflow.org/?hl=ko) : 2.6.0
* [numpy](https://docs.pyvista.org/) : 1.19.2
* [pyvista](https://docs.pyvista.org/) : 0.34.2
* [pyvistaqt](https://qtdocs.pyvista.org/) : 0.5.1
* [simpy](https://simpy.readthedocs.io/en/latest/) : 4.0.1

## Experiments
The experiment proceeds in the following order.
1. Run the simulator server
  ```
  python main_arena_server_remote_mixed.py
  ``` 
2. Connecting the client (Agent) to the server and learning the Agent
  ```
  python main_rainbow_dqn_player.py
  ``` 
And, you can also use  `main_reinforce_player_torch.py` or `main_a2c_player_torch.py` or `main_ppo_player_torch.py`.



## Result and Performance
### Success rate
<img src="https://user-images.githubusercontent.com/65005179/175872155-5a3408ba-d68f-4ecc-b149-f6977d52fc1c.png" width="60%" height="60%"/>
<img src="https://user-images.githubusercontent.com/65005179/177290231-b14e44ba-c199-477d-83b2-439f818d9073.png" width="60%" height="60%"/>


### Failure rate
<img src="https://user-images.githubusercontent.com/65005179/175872806-76ec7f55-14c9-470c-a9df-e0fe6807f5bc.png" width="60%" height="60%"/>
<img src="https://user-images.githubusercontent.com/65005179/177290898-38f5abcf-0a79-4a66-bd6d-9a39b2d6c11e.png" width="60%" height="60%"/>



