# SnakAI Project
![](https://img.shields.io/badge/Python-3.10.12-blue)
![](https://img.shields.io/badge/LICENSE-MIT-%2300557f)
![](https://img.shields.io/badge/lastest-2024--07--13-green)
![](https://img.shields.io/badge/contact-dr.mokira%40gmail.com-blueviolet)

<!-- ![](https://img.shields.io/badge/Django-5.0-%2344B78B) -->
<!-- ![](https://img.shields.io/badge/REST%20Framework-3.14.0-%23A30000) -->
<!-- ![](https://img.shields.io/badge/Swagger-OpenAPI%202.0-%23aaaa00) -->

Une application Python qui permet de gérer des tâches de manière efficace.

## Table des matières
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Fonctionnalités](#fonctionnalités)
- [Contribuer](#contribuer)
- [Licence](#licence)
- [Contact](#contact)


## Installation


```bash
git clone https://github.com/mokira3d48/SnakAI.git
cd SnakAI
python3 -m venv env
source env/bin/activate
make install
```

## Utilisation
1. `sudo apt install cmake python3-venv` Install *Cmake* and *Virtual env*;
2. `make venv` create a virtual env into directory named `env`;
3. `ssource env/bin/activate` activate the virtual environment named `env`;
4. `make install` install the requirements of this package;
5. `make dev` install the package in dev mode in virtual environment;
6. `make test` run the unit test scripts located at `tests` directory;
7. `mkae run` run script located at `src/snakai/__main__.py`.

### Exemple d'utilisation

Pour demarrer l'entrainement du modele :

```sh
make run
```

## Fonctionnalités

- Entraîner l'agent à jouer au jeux de serpent.


## Tests

Pour exécuter les tests, assurez-vous d'avoir `pytest` installé, puis exécutez :

```bash
make test  # ou pytest
```

## Contribuer

Les contributions sont les bienvenues ! Veuillez suivre ces étapes :

1. Forkez le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/ma-fonctionnalité`)
3. Commitez vos changements (`git commit -m 'Ajout d'une nouvelle fonctionnalité'`)
4. Poussez vers la branche (`git push origin feature/ma-fonctionnalité`)
5. Ouvrez une Pull Request

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## Contact

Pour toute question ou suggestion, veuillez contacter :

- **Nom** : Arnold Mokira
- **Email** : dr.mokira@gmail.com
- **GitHub** : [Votre Profil GitHub](https://github.com/mokira3d48)

```

### Explications des sections :

- **Titre et description** : Le titre du projet et une brève description de ce qu'il fait.
- **Table des matières** : Une liste de sections pour faciliter la navigation.
- **Installation** : Instructions claires sur la façon d'installer le projet.
- **Utilisation** : Exemples d'utilisation pour aider les utilisateurs à démarrer rapidement.
- **Fonctionnalités** : Une liste des fonctionnalités principales du projet.
- **Tests** : Instructions sur la façon d'exécuter les tests.
- **Contribuer** : Un guide sur la façon de contribuer au projet.
- **Licence** : Informations sur la licence du projet.
- **Contact** : Informations pour contacter le développeur ou l'équipe du projet.
```

Cet exemple de `README.md` est structuré et informatif, ce qui le rend utile
pour les utilisateurs et les contributeurs potentiels.
